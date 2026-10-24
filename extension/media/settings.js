/*
 * Settings view script.
 *
 * Initial state arrives through data-* attributes on <body> (HTML-escaped by the
 * extension), never by splicing values into this code, so a provider or model
 * name can never break the script. Secrets never travel in the other direction:
 * the extension only reports *whether* a key is saved.
 */
(function () {
  'use strict';

  var vscode = acquireVsCodeApi();
  var body = document.body;

  function readJson(value, fallback) {
    try {
      var parsed = JSON.parse(value || '');
      return parsed && typeof parsed === 'object' ? parsed : fallback;
    } catch (error) {
      return fallback;
    }
  }

  var PROVIDER_DESCRIPTIONS = {
    ollama: 'Fully local. No API key required. Models are downloaded and run on your machine.',
    openai: 'Commercial API. Requires an API key from platform.openai.com.',
    anthropic: 'Commercial API. Requires an API key from console.anthropic.com.',
    gemini: 'Google AI Studio offers a free tier. Get a key at aistudio.google.com.',
    deepseek: 'Offers a free tier. Get a key at platform.deepseek.com.',
    cohere: 'Offers a free trial tier. Get a key at dashboard.cohere.com.',
    openrouter: 'Aggregates many models. Some are free. Get a key at openrouter.ai.',
    openai_compatible: 'Use a custom endpoint (for example LM Studio, vLLM or a llama.cpp server).'
  };

  // Suggestions only: the model field accepts any name.
  var MODEL_SUGGESTIONS = {
    openai: ['gpt-4o-mini', 'gpt-4o', 'gpt-4.1', 'gpt-4.1-mini'],
    anthropic: ['claude-3-5-sonnet-20241022', 'claude-3-5-haiku-20241022'],
    gemini: ['gemini-2.0-flash', 'gemini-1.5-flash', 'gemini-1.5-pro'],
    deepseek: ['deepseek-chat', 'deepseek-coder'],
    cohere: ['command-r-plus', 'command-r'],
    openrouter: [],
    ollama: ['llama3.2'],
    openai_compatible: []
  };

  var BASE_URL_PROVIDERS = {
    ollama: {
      label: 'Ollama URL:',
      note: 'Make sure Ollama is running locally (ollama serve).',
      fallback: 'http://localhost:11434'
    },
    openai_compatible: {
      label: 'Endpoint URL:',
      note: 'Base URL of the OpenAI-compatible server, usually ending in /v1.',
      fallback: 'http://localhost:1234/v1'
    }
  };

  var state = {
    provider: body.dataset.provider || 'ollama',
    activeProvider: body.dataset.provider || 'ollama',
    models: {},
    keySaved: readJson(body.dataset.keyStates, {}),
    baseUrls: readJson(body.dataset.baseUrls, {}),
    suggestions: {},
    busy: false
  };
  state.models[state.provider] = body.dataset.model || '';

  function $(id) {
    return document.getElementById(id);
  }

  function needsKey(provider) {
    return provider !== 'ollama';
  }

  function show(el, visible) {
    el.style.display = visible ? '' : 'none';
  }

  var statusTimer = null;
  function setStatus(message, isError) {
    var el = $('statusMessage');
    if (statusTimer) {
      clearTimeout(statusTimer);
      statusTimer = null;
    }
    if (!message) {
      show(el, false);
      return;
    }
    el.textContent = message;
    el.className = 'status ' + (isError ? 'error' : 'ok');
    show(el, true);
    if (!isError) {
      statusTimer = setTimeout(function () {
        show(el, false);
      }, 8000);
    }
  }

  function setBusy(busy) {
    state.busy = busy;
    ['saveBtn', 'testBtn', 'removeKeyBtn', 'refreshModelsBtn'].forEach(function (id) {
      $(id).disabled = busy;
    });
  }

  function modelSuggestions(provider) {
    return (state.suggestions[provider] || MODEL_SUGGESTIONS[provider] || []).slice();
  }

  function fillModelList(provider) {
    var list = $('modelList');
    list.textContent = '';
    modelSuggestions(provider).forEach(function (name) {
      var option = document.createElement('option');
      option.value = name;
      list.appendChild(option);
    });
  }

  function currentBaseUrl(provider) {
    return state.baseUrls[provider] || (BASE_URL_PROVIDERS[provider] || {}).fallback || '';
  }

  function render() {
    var provider = state.provider;
    $('providerSelect').value = provider;
    $('providerDescription').textContent = PROVIDER_DESCRIPTIONS[provider] || '';

    // Model: remember what was typed per provider; otherwise suggest a default
    // that actually belongs to the chosen provider (never another provider's).
    var model = state.models[provider];
    if (model === undefined) {
      model = modelSuggestions(provider)[0] || '';
      state.models[provider] = model;
    }
    $('modelInput').value = model;
    fillModelList(provider);

    var keyRequired = needsKey(provider);
    show($('apiKeySection'), keyRequired);
    show($('removeKeyBtn'), keyRequired && !!state.keySaved[provider]);
    var keyState = $('keyState');
    if (keyRequired) {
      if (state.keySaved[provider]) {
        keyState.textContent =
          '✓ An API key is saved for this provider (VS Code SecretStorage). Enter a new key only to replace it.';
        keyState.className = 'key-state saved';
        $('apiKeyInput').placeholder = 'Saved — enter a new key to replace it';
      } else {
        keyState.textContent = 'No API key saved for this provider yet.';
        keyState.className = 'key-state missing';
        $('apiKeyInput').placeholder = 'Enter API key';
      }
    }

    var baseInfo = BASE_URL_PROVIDERS[provider];
    show($('baseUrlSection'), !!baseInfo);
    if (baseInfo) {
      $('baseUrlLabel').textContent = baseInfo.label;
      $('baseUrlNote').textContent = baseInfo.note;
      $('baseUrlInput').value = currentBaseUrl(provider);
    }
  }

  // Remember edits so switching provider back and forth does not lose them.
  function rememberInputs() {
    state.models[state.provider] = $('modelInput').value;
    if (BASE_URL_PROVIDERS[state.provider]) {
      state.baseUrls[state.provider] = $('baseUrlInput').value;
    }
  }

  function collect() {
    rememberInputs();
    var provider = state.provider;
    var payload = {
      provider: provider,
      model: $('modelInput').value.trim()
    };
    var key = $('apiKeyInput').value.trim();
    if (needsKey(provider) && key) {
      payload.apiKey = key;
    }
    if (BASE_URL_PROVIDERS[provider]) {
      payload.baseUrl = $('baseUrlInput').value.trim();
    }
    return payload;
  }

  function onSave() {
    if (state.busy) {
      return;
    }
    var payload = collect();
    if (!payload.model) {
      setStatus('Enter a model name before saving.', true);
      return;
    }
    setBusy(true);
    setStatus('Saving…', false);
    payload.command = 'saveConfiguration';
    vscode.postMessage(payload);
  }

  function onTest() {
    if (state.busy) {
      return;
    }
    var payload = collect();
    setBusy(true);
    setStatus('Testing connection…', false);
    payload.command = 'testConnection';
    vscode.postMessage(payload);
  }

  function onRemoveKey() {
    if (state.busy) {
      return;
    }
    setBusy(true);
    setStatus('Removing API key…', false);
    vscode.postMessage({ command: 'removeApiKey', provider: state.provider });
  }

  function onRefreshModels() {
    if (state.busy) {
      return;
    }
    setBusy(true);
    setStatus('Loading models…', false);
    vscode.postMessage({ command: 'refreshModels', provider: state.provider });
  }

  window.addEventListener('message', function (event) {
    var message = event.data;
    if (!message || typeof message !== 'object') {
      return;
    }
    switch (message.command) {
      case 'showStatus':
        setStatus(String(message.text || ''), !!message.isError);
        break;
      case 'idle':
        setBusy(false);
        break;
      case 'keyState':
        if (typeof message.provider === 'string') {
          state.keySaved[message.provider] = !!message.saved;
          if (message.provider === state.provider) {
            $('apiKeyInput').value = '';
            render();
          }
        }
        break;
      case 'models':
        if (typeof message.provider === 'string' && Array.isArray(message.models)) {
          state.suggestions[message.provider] = message.models.filter(function (m) {
            return typeof m === 'string';
          });
          if (message.provider === state.provider) {
            fillModelList(state.provider);
          }
        }
        break;
      default:
        break;
    }
  });

  function init() {
    try {
      render();
      $('providerSelect').addEventListener('change', function () {
        rememberInputs();
        $('apiKeyInput').value = '';
        state.provider = this.value;
        setStatus('', false);
        render();
      });
      $('saveBtn').addEventListener('click', onSave);
      $('testBtn').addEventListener('click', onTest);
      $('removeKeyBtn').addEventListener('click', onRemoveKey);
      $('refreshModelsBtn').addEventListener('click', onRefreshModels);
      vscode.postMessage({ command: 'ready' });
    } catch (error) {
      setStatus('Settings failed to initialize. Reload the window; the extension log has details.', true);
    }
  }

  init();
})();
