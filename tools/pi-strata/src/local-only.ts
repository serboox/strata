import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";

export const LOCAL_ONLY_NOTICE = "this sandbox runs local models only";
export const LOCAL_ONLY_ENV = "SPI_LOCAL_ONLY";

export function localOnlyRequested(env: Record<string, string | undefined>): boolean {
  return env[LOCAL_ONLY_ENV] === "1";
}

const LOOPBACK_HOSTS = new Set(["127.0.0.1", "localhost", "[::1]"]);
const REFUSED_COMMAND = /^\/(login|logout)(?:\s|$)/i;
const MODEL_KINDS = ["chat", "image", "classifier"] as const;

export type ModelKind = (typeof MODEL_KINDS)[number];

export interface ModelLike {
  provider: string;
  id: string;
  baseUrl?: string;
}

export interface RegistryLike {
  getModelsOfType(kind: ModelKind, provider?: string): readonly ModelLike[];
  getRegisteredProviderIds(): readonly string[];
  registerProvider(name: string, config: { models: [] }): void;
  unregisterProvider(name: string): void;
}

export interface SweepOutcome {
  removed: string[];
  failed: string[];
}

// A model is local only when its endpoint is a loopback address; a missing or unparsable address counts as external.
export function isLocalUrl(url: unknown): boolean {
  if (typeof url !== "string") return false;
  try {
    const { protocol, hostname } = new URL(url);
    return (protocol === "http:" || protocol === "https:") && LOOPBACK_HOSTS.has(hostname);
  } catch {
    return false;
  }
}

export function isLocalModel(model: ModelLike | undefined): boolean {
  return model !== undefined && isLocalUrl(model.baseUrl);
}

// The command a submitted editor text would run, when it is one that connects or disconnects a provider.
export function refusedCommand(editorText: string): string | undefined {
  return REFUSED_COMMAND.exec(editorText.trim())?.[1]?.toLowerCase();
}

function modelsByProvider(registry: RegistryLike): Map<string, ModelLike[]> {
  const byProvider = new Map<string, ModelLike[]>();
  for (const kind of MODEL_KINDS) {
    for (const model of registry.getModelsOfType(kind)) {
      const models = byProvider.get(model.provider) ?? [];
      models.push(model);
      byProvider.set(model.provider, models);
    }
  }
  for (const provider of registry.getRegisteredProviderIds()) {
    if (!byProvider.has(provider)) byProvider.set(provider, []);
  }
  return byProvider;
}

export class CatalogGuard {
  private readonly emptied = new Set<string>();

  // Drops every provider that is not entirely local. A provider with one external model goes whole: the registry cannot
  // filter a single model out of a provider.
  sweep(registry: RegistryLike): SweepOutcome {
    const outcome: SweepOutcome = { removed: [], failed: [] };
    for (const [provider, models] of modelsByProvider(registry)) {
      if (models.length === 0 && this.emptied.has(provider)) continue;
      if (models.length > 0 && models.every(isLocalModel)) continue;
      try {
        this.drop(registry, provider);
        outcome.removed.push(provider);
      } catch {
        outcome.failed.push(provider);
      }
    }
    return outcome;
  }

  // Unregistering removes what an extension added; a built-in or models.json provider comes back and is emptied instead.
  private drop(registry: RegistryLike, provider: string): void {
    registry.unregisterProvider(provider);
    if (!MODEL_KINDS.some((kind) => registry.getModelsOfType(kind, provider).length > 0)) return;
    registry.registerProvider(provider, { models: [] });
    this.emptied.add(provider);
  }
}

function blocked(ctx: ExtensionContext): boolean {
  try {
    return ctx.model !== undefined && !isLocalModel(ctx.model);
  } catch {
    return true;
  }
}

export interface LocalOnlyOptions {
  isSubmitKey: (data: string) => boolean;
}

export function installLocalOnly(pi: ExtensionAPI, options: LocalOnlyOptions): void {
  const catalog = new CatalogGuard();
  let stopWatchingSubmit: (() => void) | undefined;

  const tell = (ctx: ExtensionContext, message: string, level: "warning" | "error" = "warning") => {
    if (ctx.hasUI) ctx.ui.notify(`strata: ${message}`, level);
    else console.error(`strata: ${message}`);
  };

  const sweep = (ctx: ExtensionContext, announce: boolean) => {
    let outcome: SweepOutcome;
    try {
      outcome = catalog.sweep(ctx.modelRegistry);
    } catch {
      tell(ctx, `the provider catalog could not be checked; requests to a non-local model are still refused because ${LOCAL_ONLY_NOTICE}`, "error");
      return;
    }
    if (outcome.failed.length > 0) tell(ctx, `could not remove the external providers ${outcome.failed.join(", ")}; they stay unusable because ${LOCAL_ONLY_NOTICE}`, "error");
    if (announce && outcome.removed.length > 0) tell(ctx, `removed the external providers ${outcome.removed.join(", ")}: ${LOCAL_ONLY_NOTICE}`);
  };

  const returnToLocal = async (ctx: ExtensionContext, previous?: ModelLike): Promise<boolean> => {
    const candidates = [previous, ...ctx.modelRegistry.getAvailable()].filter((model) => isLocalModel(model));
    for (const model of candidates) {
      if (await pi.setModel(model as never)) return true;
    }
    return false;
  };

  const watchSubmit = (ctx: ExtensionContext) => {
    stopWatchingSubmit?.();
    stopWatchingSubmit = undefined;
    if (ctx.mode !== "tui") return;
    stopWatchingSubmit = ctx.ui.onTerminalInput((data) => {
      if (!options.isSubmitKey(data)) return undefined;
      sweep(ctx, true);
      try {
        const command = refusedCommand(ctx.ui.getEditorText());
        if (command === undefined) return undefined;
        ctx.ui.setEditorText("");
        ctx.ui.notify(`strata: /${command} is disabled: ${LOCAL_ONLY_NOTICE}.`, "warning");
      } catch {
        // The editor text cannot be read, so a login command cannot be ruled out: the key is dropped.
      }
      return { consume: true };
    });
  };

  pi.on("session_start", async (_event, ctx) => {
    watchSubmit(ctx);
    sweep(ctx, false);
    if (blocked(ctx) && !(await returnToLocal(ctx))) tell(ctx, `the active model is not local and no local model could be selected; requests are refused because ${LOCAL_ONLY_NOTICE}`, "error");
  });

  // The working folder is writable from the sandbox, so a project extension could be planted there and register a provider.
  pi.on("project_trust", () => ({ trusted: "no" as const }));

  pi.on("session_shutdown", () => {
    stopWatchingSubmit?.();
    stopWatchingSubmit = undefined;
  });

  pi.on("model_select", async (event, ctx) => {
    sweep(ctx, true);
    if (isLocalModel(event.model)) return;
    const switched = await returnToLocal(ctx, event.previousModel);
    tell(ctx, `${event.model.provider}/${event.model.id} is not a local model: ${switched ? "the previous model is back" : "no local model could be selected, requests stay refused"}`);
  });

  pi.on("input", (_event, ctx) => {
    sweep(ctx, true);
    if (!blocked(ctx)) return undefined;
    tell(ctx, `the active model is not local, the message was not sent: ${LOCAL_ONLY_NOTICE}`);
    return { action: "handled" as const };
  });

  pi.on("before_provider_request", (_event, ctx) => {
    if (!blocked(ctx)) return undefined;
    ctx.abort();
    tell(ctx, `a request to a non-local model was stopped: ${LOCAL_ONLY_NOTICE}`);
    return {};
  });

  pi.on("session_before_compact", (_event, ctx) => (blocked(ctx) ? { cancel: true } : undefined));
}
