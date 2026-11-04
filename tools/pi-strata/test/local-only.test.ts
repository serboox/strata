import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { test } from "node:test";
import { pathToFileURL } from "node:url";
import { CatalogGuard, installLocalOnly, isLocalModel, isLocalUrl, localOnlyRequested, refusedCommand, type ModelKind, type ModelLike } from "../src/local-only.ts";

const LOCAL_URL = "http://127.0.0.1:8080/v1";
const EXTERNAL_URL = "https://api.example.com/v1";

interface FakeModel extends ModelLike {
  kind: ModelKind;
}

function model(provider: string, id: string, baseUrl: string | undefined, kind: ModelKind = "chat"): FakeModel {
  return { provider, id, baseUrl, kind };
}

class FakeRegistry {
  readonly calls: string[] = [];
  readonly failing = new Set<string>();
  private readonly builtin = new Map<string, FakeModel[]>();
  private readonly registered = new Map<string, FakeModel[]>();

  constructor(builtin: Record<string, FakeModel[]>) {
    for (const [provider, models] of Object.entries(builtin)) this.builtin.set(provider, models);
  }

  // What another extension would do: a provider that brings its own models.
  addExtensionProvider(provider: string, models: FakeModel[]): void {
    this.registered.set(provider, models);
  }

  providers(): string[] {
    return [...new Set([...this.builtin.keys(), ...this.registered.keys()])];
  }

  getModelsOfType(kind: ModelKind, provider?: string): readonly FakeModel[] {
    const providers = provider === undefined ? this.providers() : [provider];
    return providers.flatMap((id) => this.registered.get(id) ?? this.builtin.get(id) ?? []).filter((entry) => entry.kind === kind);
  }

  getAvailable(): readonly FakeModel[] {
    return this.getModelsOfType("chat");
  }

  getRegisteredProviderIds(): readonly string[] {
    return [...this.registered.keys()];
  }

  registerProvider(name: string, config: { models: [] }): void {
    this.calls.push(`register ${name}`);
    if (this.failing.has(name)) throw new Error("registry failure");
    assert.deepEqual(config, { models: [] });
    this.registered.set(name, []);
  }

  unregisterProvider(name: string): void {
    this.calls.push(`unregister ${name}`);
    if (this.failing.has(name)) throw new Error("registry failure");
    this.registered.delete(name);
  }

  remainingModels(): string[] {
    return (["chat", "image", "classifier"] as const).flatMap((kind) => this.getModelsOfType(kind).map((entry) => `${entry.provider}/${entry.id}`));
  }
}

function catalog(): FakeRegistry {
  return new FakeRegistry({
    strata: [model("strata", "local-model", LOCAL_URL)],
    anthropic: [model("anthropic", "big", "https://api.anthropic.com")],
    openai: [model("openai", "gpt", "https://api.openai.com/v1"), model("openai", "gpt-image", "https://api.openai.com/v1", "image")],
    "amazon-bedrock": [model("amazon-bedrock", "claude", undefined)],
    typesafe: [model("typesafe", "router", "https://api.typesafe.example", "classifier")],
  });
}

test("only loopback http(s) addresses are local", () => {
  for (const url of ["http://127.0.0.1:8080/v1", "http://localhost:8080", "https://localhost/v1", "http://[::1]:8080/v1", "HTTP://LOCALHOST/v1"]) {
    assert.equal(isLocalUrl(url), true, url);
  }
  for (const url of [
    "https://api.anthropic.com",
    "http://127.0.0.1@evil.example/v1",
    "http://127.0.0.1.evil.example/v1",
    "http://localhost.evil.example/v1",
    "http://evil.example/127.0.0.1",
    "ftp://127.0.0.1/v1",
    "file:///127.0.0.1",
    "not a url",
    "",
    undefined,
    null,
    42,
  ]) {
    assert.equal(isLocalUrl(url), false, String(url));
  }
});

test("a model without an address is external", () => {
  assert.equal(isLocalModel(model("strata", "a", LOCAL_URL)), true);
  assert.equal(isLocalModel(model("amazon-bedrock", "a", undefined)), false);
  assert.equal(isLocalModel(undefined), false);
});

test("refusedCommand names the login commands and nothing else", () => {
  assert.equal(refusedCommand("/login"), "login");
  assert.equal(refusedCommand("/login anthropic"), "login");
  assert.equal(refusedCommand("  /logout  "), "logout");
  assert.equal(refusedCommand("/LOGIN"), "login");
  assert.equal(refusedCommand("/login\tradius"), "login");
  for (const text of ["", "/loginx", "/logoutnow", "/model", "/model login", "please /login", "login", "/ login"]) {
    assert.equal(refusedCommand(text), undefined, text);
  }
});

test("the sweep empties every external provider of every model kind and keeps the local one", () => {
  const registry = catalog();
  const outcome = new CatalogGuard().sweep(registry);
  assert.deepEqual(outcome.failed, []);
  assert.deepEqual(outcome.removed.sort(), ["amazon-bedrock", "anthropic", "openai", "typesafe"]);
  assert.deepEqual(registry.remainingModels(), ["strata/local-model"]);
});

test("a second sweep finds nothing to do", () => {
  const registry = catalog();
  const guard = new CatalogGuard();
  guard.sweep(registry);
  registry.calls.length = 0;
  assert.deepEqual(guard.sweep(registry), { removed: [], failed: [] });
  assert.deepEqual(registry.calls, []);
});

test("a provider with one external model among local ones is dropped whole", () => {
  const registry = new FakeRegistry({ mixed: [model("mixed", "near", LOCAL_URL), model("mixed", "far", EXTERNAL_URL)] });
  new CatalogGuard().sweep(registry);
  assert.deepEqual(registry.remainingModels(), []);
});

test("an external provider registered by another extension is removed", () => {
  const registry = catalog();
  const guard = new CatalogGuard();
  guard.sweep(registry);
  registry.addExtensionProvider("shady", [model("shady", "m", EXTERNAL_URL)]);
  assert.deepEqual(guard.sweep(registry).removed, ["shady"]);
  assert.deepEqual(registry.remainingModels(), ["strata/local-model"]);
  assert.deepEqual(registry.getRegisteredProviderIds().filter((id) => id === "shady"), []);
});

test("an extension that re-registers a built-in provider with its own external models is swept again", () => {
  const registry = catalog();
  const guard = new CatalogGuard();
  guard.sweep(registry);
  registry.addExtensionProvider("anthropic", [model("anthropic", "proxy", EXTERNAL_URL)]);
  assert.deepEqual(guard.sweep(registry).removed, ["anthropic"]);
  assert.deepEqual(registry.remainingModels(), ["strata/local-model"]);
});

test("an extension provider with no models yet, such as one waiting for a login, is removed", () => {
  const registry = catalog();
  registry.addExtensionProvider("waiting", []);
  assert.deepEqual(new CatalogGuard().sweep(registry).removed.includes("waiting"), true);
  assert.deepEqual(registry.getRegisteredProviderIds().includes("waiting"), false);
});

test("a local provider registered by an extension keeps working", () => {
  const registry = catalog();
  registry.addExtensionProvider("second-local", [model("second-local", "tiny", "http://localhost:9000/v1")]);
  new CatalogGuard().sweep(registry);
  assert.deepEqual(registry.remainingModels().sort(), ["second-local/tiny", "strata/local-model"]);
});

test("a provider that cannot be removed is reported and does not stop the sweep", () => {
  const registry = catalog();
  registry.failing.add("anthropic");
  const outcome = new CatalogGuard().sweep(registry);
  assert.deepEqual(outcome.failed, ["anthropic"]);
  assert.deepEqual(outcome.removed.sort(), ["amazon-bedrock", "openai", "typesafe"]);
});

interface Notice {
  message: string;
  level: string | undefined;
}

type Handler = (event: unknown, ctx: unknown) => unknown;

function host(options: { mode?: string; model?: ModelLike | undefined; registry?: FakeRegistry; refuseSetModel?: boolean } = {}) {
  const registry = options.registry ?? catalog();
  const handlers = new Map<string, Handler[]>();
  const notices: Notice[] = [];
  const submitListeners: ((data: string) => { consume?: boolean } | undefined)[] = [];
  const setModelCalls: ModelLike[] = [];
  const state = { model: "model" in options ? options.model : registry.getModelsOfType("chat")[0], editorText: "", aborts: 0 };

  const pi = {
    on: (event: string, handler: Handler) => {
      handlers.set(event, [...(handlers.get(event) ?? []), handler]);
    },
    setModel: async (selected: ModelLike) => {
      setModelCalls.push(selected);
      if (options.refuseSetModel) return false;
      state.model = selected;
      return true;
    },
  };
  const ctx = {
    mode: options.mode ?? "tui",
    hasUI: true,
    get model() {
      return state.model;
    },
    modelRegistry: registry,
    abort: () => {
      state.aborts++;
    },
    ui: {
      notify: (message: string, level?: string) => notices.push({ message, level }),
      getEditorText: () => state.editorText,
      setEditorText: (text: string) => {
        state.editorText = text;
      },
      onTerminalInput: (listener: (data: string) => { consume?: boolean } | undefined) => {
        submitListeners.push(listener);
        return () => {
          submitListeners.splice(submitListeners.indexOf(listener), 1);
        };
      },
    },
  };
  installLocalOnly(pi as never, { isSubmitKey: (data) => data === "\r" });

  const fire = async (event: string, payload: unknown = {}) => {
    let result: unknown;
    for (const handler of handlers.get(event) ?? []) result = (await handler({ type: event, ...(payload as object) }, ctx)) ?? result;
    return result;
  };
  const press = (data: string) => {
    for (const listener of [...submitListeners]) {
      const result = listener(data);
      if (result?.consume) return result;
    }
    return undefined;
  };
  return { registry, notices, setModelCalls, state, fire, press, listeners: submitListeners };
}

const LOCAL_MODEL = model("strata", "local-model", LOCAL_URL);
const EXTERNAL_MODEL = model("anthropic", "big", "https://api.anthropic.com");

test("at start the external providers are gone and the local model stays selected", async () => {
  const { registry, fire, setModelCalls, notices } = host({ model: LOCAL_MODEL });
  await fire("session_start");
  assert.deepEqual(registry.remainingModels(), ["strata/local-model"]);
  assert.deepEqual(setModelCalls, []);
  assert.deepEqual(notices, []);
});

test("a session that starts on an external model is moved to a local one", async () => {
  const { fire, setModelCalls, state } = host({ model: EXTERNAL_MODEL });
  await fire("session_start");
  assert.equal(setModelCalls.length, 1);
  assert.equal(state.model?.provider, "strata");
});

test("when no local model can be selected the session says so and requests stay refused", async () => {
  const { fire, notices } = host({ model: EXTERNAL_MODEL, refuseSetModel: true });
  await fire("session_start");
  assert.match(notices.at(-1)?.message ?? "", /no local model could be selected/);
  assert.deepEqual(await fire("input", { text: "hello", source: "interactive" }), { action: "handled" });
});

test("the login guard is in place even when the switch to a local model fails at start", async () => {
  const registry = catalog();
  registry.getAvailable = () => {
    throw new Error("registry failure");
  };
  const { fire, listeners } = host({ model: EXTERNAL_MODEL, registry });
  await assert.rejects(fire("session_start"), /registry failure/);
  assert.equal(listeners.length, 1);
});

test("/login and /logout are refused with a message and never reach the editor", async () => {
  const { fire, press, state, notices } = host({ model: LOCAL_MODEL });
  await fire("session_start");
  for (const command of ["/login", "/login anthropic", "/logout"]) {
    notices.length = 0;
    state.editorText = command;
    assert.deepEqual(press("\r"), { consume: true }, command);
    assert.equal(state.editorText, "");
    assert.match(notices[0]?.message ?? "", new RegExp(`${command.split(" ")[0]} is disabled: this sandbox runs local models only`));
    assert.equal(notices[0]?.level, "warning");
  }
});

test("a provider that appears after the start is gone before the model list is opened", async () => {
  const { registry, fire, press, state, notices } = host({ model: LOCAL_MODEL });
  await fire("session_start");
  registry.addExtensionProvider("shady", [model("shady", "m", EXTERNAL_URL)]);
  state.editorText = "/model";
  assert.equal(press("\r"), undefined);
  assert.deepEqual(registry.remainingModels(), ["strata/local-model"]);
  assert.match(notices.at(-1)?.message ?? "", /removed the external providers shady/);
});

test("other commands, ordinary text and other keys are not touched by the login guard", async () => {
  const { fire, press, state, notices } = host({ model: LOCAL_MODEL });
  await fire("session_start");
  for (const text of ["/model", "/scoped-models", "explain /login in pi", ""]) {
    state.editorText = text;
    assert.equal(press("\r"), undefined, text);
    assert.equal(state.editorText, text);
  }
  state.editorText = "/login";
  assert.equal(press("x"), undefined);
  assert.equal(state.editorText, "/login");
  assert.deepEqual(notices, []);
});

test("when the editor text cannot be read the submit key is dropped", async () => {
  const { fire, press, state } = host({ model: LOCAL_MODEL });
  await fire("session_start");
  Object.defineProperty(state, "editorText", {
    get: () => {
      throw new Error("editor is gone");
    },
  });
  assert.deepEqual(press("\r"), { consume: true });
});

test("the login guard is installed only where there is a terminal, and removed at shutdown", async () => {
  const printing = host({ mode: "print", model: LOCAL_MODEL });
  await printing.fire("session_start");
  assert.equal(printing.listeners.length, 0);

  const terminal = host({ model: LOCAL_MODEL });
  await terminal.fire("session_start");
  await terminal.fire("session_start");
  assert.equal(terminal.listeners.length, 1);
  await terminal.fire("session_shutdown");
  assert.equal(terminal.listeners.length, 0);
});

test("selecting an external model puts the previous local model back", async () => {
  const { fire, state, setModelCalls, notices } = host({ model: EXTERNAL_MODEL });
  await fire("model_select", { model: EXTERNAL_MODEL, previousModel: LOCAL_MODEL, source: "set" });
  assert.deepEqual(setModelCalls, [LOCAL_MODEL]);
  assert.equal(state.model, LOCAL_MODEL);
  assert.match(notices.at(-1)?.message ?? "", /anthropic\/big is not a local model: the previous model is back/);
});

test("selecting a local model is accepted without a word", async () => {
  const { fire, setModelCalls, notices } = host({ model: LOCAL_MODEL });
  await fire("session_start");
  await fire("model_select", { model: LOCAL_MODEL, previousModel: undefined, source: "cycle" });
  assert.deepEqual(setModelCalls, []);
  assert.deepEqual(notices, []);
});

test("a message is not sent while the active model is not local", async () => {
  const { fire, notices } = host({ model: EXTERNAL_MODEL });
  assert.deepEqual(await fire("input", { text: "hello", source: "interactive" }), { action: "handled" });
  assert.match(notices.at(-1)?.message ?? "", /the message was not sent: this sandbox runs local models only/);
});

test("a message goes through while the active model is local, and when there is no model", async () => {
  assert.equal(await host({ model: LOCAL_MODEL }).fire("input", { text: "hello", source: "interactive" }), undefined);
  assert.equal(await host({ model: undefined }).fire("input", { text: "hello", source: "interactive" }), undefined);
});

test("a provider registered by an extension after the start is removed on the next message", async () => {
  const { registry, fire } = host({ model: LOCAL_MODEL });
  await fire("session_start");
  registry.addExtensionProvider("shady", [model("shady", "m", EXTERNAL_URL)]);
  assert.equal(await fire("input", { text: "hello", source: "interactive" }), undefined);
  assert.deepEqual(registry.remainingModels(), ["strata/local-model"]);
});

test("a request that reaches the provider layer with an external model is aborted and emptied", async () => {
  const external = host({ model: EXTERNAL_MODEL });
  assert.deepEqual(await external.fire("before_provider_request", { payload: { messages: ["secret"] } }), {});
  assert.equal(external.state.aborts, 1);

  const local = host({ model: LOCAL_MODEL });
  assert.equal(await local.fire("before_provider_request", { payload: { messages: ["ok"] } }), undefined);
  assert.equal(local.state.aborts, 0);
});

test("a request is refused when the active model cannot even be read", async () => {
  const { fire, state } = host({ model: LOCAL_MODEL });
  Object.defineProperty(state, "model", { get: () => { throw new Error("stale context"); }, configurable: true });
  assert.deepEqual(await fire("before_provider_request", { payload: {} }), {});
});

test("compaction is cancelled on an external model and left alone on a local one", async () => {
  assert.deepEqual(await host({ model: EXTERNAL_MODEL }).fire("session_before_compact"), { cancel: true });
  assert.equal(await host({ model: LOCAL_MODEL }).fire("session_before_compact"), undefined);
});

test("project extensions are not trusted, so they cannot register providers", async () => {
  assert.deepEqual(await host({ model: LOCAL_MODEL }).fire("project_trust", { cwd: "/work/project" }), { trusted: "no" });
});

const piEntry = join(dirname(dirname(process.execPath)), "lib/node_modules/@earendil-works/pi-coding-agent/dist/index.js");

test("against the real pi model registry nothing but the local provider is left", { skip: !existsSync(piEntry) && "pi is not installed next to node" }, async () => {
  const saved = { offline: process.env.PI_OFFLINE, key: process.env.OPENAI_API_KEY };
  process.env.PI_OFFLINE = "1";
  process.env.OPENAI_API_KEY = "sk-test-not-real";
  const dir = await mkdtemp(join(tmpdir(), "strata-local-only-"));
  try {
    const cost = { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 };
    const definition = { name: "Local", reasoning: false, input: ["text"], contextWindow: 65536, maxTokens: 8192, cost };
    await writeFile(
      join(dir, "models.json"),
      JSON.stringify({
        providers: {
          strata: { baseUrl: LOCAL_URL, api: "openai-completions", apiKey: "none", models: [{ id: "local-model", ...definition }] },
          corp: { baseUrl: EXTERNAL_URL, api: "openai-completions", apiKey: "none", models: [{ id: "corp-model", ...definition }] },
        },
      }),
    );
    await writeFile(join(dir, "auth.json"), "{}");
    const pi = (await import(pathToFileURL(piEntry).href)) as typeof import("@earendil-works/pi-coding-agent");
    const runtime = await pi.ModelRuntime.create({ modelsPath: join(dir, "models.json"), authPath: join(dir, "auth.json"), modelsStorePath: join(dir, "store.json") });
    const registry = new pi.ModelRegistry(runtime);

    const before = new Set((["chat", "image", "classifier"] as const).flatMap((kind) => registry.getModelsOfType(kind).map((entry) => entry.provider)));
    assert.ok(before.size > 5, "the built-in catalog lists many providers");
    assert.ok(registry.getAvailable().some((entry) => entry.provider === "openai"), "an environment key makes a built-in provider usable");
    assert.ok(registry.find("corp", "corp-model"), "a provider of models.json is listed before the sweep");

    const guard = new CatalogGuard();
    const outcome = guard.sweep(registry as never);
    await registry.refresh({ allowNetwork: false });
    assert.deepEqual(outcome.failed, []);
    for (const provider of ["anthropic", "openai", "typesafe", "corp"]) assert.ok(outcome.removed.includes(provider), provider);

    const left = (["chat", "image", "classifier"] as const).flatMap((kind) => registry.getModelsOfType(kind).map((entry) => `${entry.provider}/${entry.id}`));
    assert.deepEqual(left, ["strata/local-model"]);
    assert.deepEqual(registry.getAvailable().map((entry) => `${entry.provider}/${entry.id}`), ["strata/local-model"]);
    assert.equal(registry.find("strata", "local-model")?.baseUrl, LOCAL_URL);
    assert.equal(registry.find("openai", "gpt-5"), undefined);

    registry.registerProvider("shady", {
      baseUrl: EXTERNAL_URL,
      apiKey: "x",
      api: "openai-completions",
      models: [{ id: "m", name: "M", reasoning: false, input: ["text"], contextWindow: 1000, maxTokens: 100, cost }],
    });
    assert.ok(registry.find("shady", "m"), "the hypothetical provider is registered before the sweep");
    assert.deepEqual(guard.sweep(registry as never).removed, ["shady"]);
    await registry.refresh({ allowNetwork: false });
    assert.equal(registry.find("shady", "m"), undefined);
    assert.deepEqual(registry.getRegisteredProviderIds().filter((id) => id === "shady"), []);

    registry.registerProvider("sso", {
      baseUrl: EXTERNAL_URL,
      api: "openai-completions",
      models: [{ id: "m", name: "M", reasoning: false, input: ["text"], contextWindow: 1000, maxTokens: 100, cost }],
      oauth: {
        name: "Corporate SSO",
        login: async () => ({ refresh: "r", access: "a", expires: Date.now() + 60_000 }),
        refreshToken: async (credentials) => credentials,
        getApiKey: (credentials) => credentials.access,
      },
    });
    assert.ok(runtime.getProviders().some((provider) => provider.id === "sso" && provider.auth.oauth !== undefined), "the hypothetical provider offers a login before the sweep");
    assert.deepEqual(guard.sweep(registry as never).removed, ["sso"]);
    assert.equal(runtime.getProviders().some((provider) => provider.id === "sso"), false);

    assert.deepEqual(guard.sweep(registry as never), { removed: [], failed: [] });
  } finally {
    await rm(dir, { recursive: true, force: true });
    if (saved.offline === undefined) delete process.env.PI_OFFLINE;
    else process.env.PI_OFFLINE = saved.offline;
    if (saved.key === undefined) delete process.env.OPENAI_API_KEY;
    else process.env.OPENAI_API_KEY = saved.key;
  }
});

test("localOnlyRequested is on only for the exact value 1", () => {
  assert.equal(localOnlyRequested({ SPI_LOCAL_ONLY: "1" }), true);
  for (const value of [undefined, "", "0", "true", "yes", " 1"]) assert.equal(localOnlyRequested({ SPI_LOCAL_ONLY: value }), false, String(value));
  assert.equal(localOnlyRequested({}), false);
});
