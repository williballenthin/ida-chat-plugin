/**
 * IDA Chat Agent Harness - Pure JS, zero npm dependencies at runtime.
 *
 * Designed to run in PythonMonkey's SpiderMonkey engine. All I/O is
 * delegated to Python callbacks:
 *   - pythonFetch(url, method, headersJson, body) -> responseJson string
 *   - pythonToolCallback(code) -> result string
 *
 * Uses OpenAI-compatible API format (works with OpenRouter and others).
 */

// Module state (set by init())
var config = null;
var toolCallback = null;
var fetchFn = null;
var messages = [];
var eventLog = [];

/**
 * Known provider base URLs.
 */
var PROVIDERS = {
  openrouter: { baseUrl: "https://openrouter.ai/api/v1", envKey: "OPENROUTER_API_KEY" },
  openai: { baseUrl: "https://api.openai.com/v1", envKey: "OPENAI_API_KEY" },
  groq: { baseUrl: "https://api.groq.com/openai/v1", envKey: "GROQ_API_KEY" },
  cerebras: { baseUrl: "https://api.cerebras.ai/v1", envKey: "CEREBRAS_API_KEY" },
  xai: { baseUrl: "https://api.x.ai/v1", envKey: "XAI_API_KEY" },
};

/**
 * The ida_script tool definition in OpenAI tool format.
 */
var IDA_SCRIPT_TOOL = {
  type: "function",
  "function": {
    name: "ida_script",
    description:
      "Execute Python analysis code against an IDA Pro database. " +
      "Available functions: get_binary_info(), enumerate_functions(), get_function_by_name(name), " +
      "disassemble_function(addr), decompile_function(addr), get_function_signature(addr), " +
      "get_callers(addr), get_callees(addr), get_basic_blocks(addr), " +
      "get_xrefs_to(addr), get_xrefs_from(addr), enumerate_strings(), get_string_at(addr), " +
      "enumerate_segments(), enumerate_names(), get_name_at(addr), demangle_name(name), " +
      "enumerate_imports(), enumerate_entries(), read_bytes(addr, size), find_bytes(pattern), " +
      "get_disassembly_at(addr), get_instruction_at(addr), is_code_at(addr), is_data_at(addr), " +
      "is_valid_address(addr), get_comment_at(addr). Use print() to output results.",
    parameters: {
      type: "object",
      required: ["code"],
      properties: {
        code: { type: "string", description: "Python code to execute against the IDA database." },
      },
    },
  },
};

/**
 * Initialize the agent.
 */
function init(systemPrompt, modelStr, pythonToolCb, pythonFetchFn, apiKey) {
  toolCallback = pythonToolCb;
  fetchFn = pythonFetchFn;

  var slashIdx = modelStr.indexOf("/");
  var providerName = modelStr.substring(0, slashIdx);
  var modelId = modelStr.substring(slashIdx + 1);

  var provider = PROVIDERS[providerName] || PROVIDERS.openrouter;

  config = {
    provider: providerName,
    modelId: modelId,
    baseUrl: provider.baseUrl,
    apiKey: apiKey || "",
    systemPrompt: systemPrompt,
  };

  messages = [];
  eventLog = [];
}

/**
 * Make an LLM API call via Python fetch callback.
 */
function callLLM(msgs) {
  var url = config.baseUrl + "/chat/completions";
  var headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer " + config.apiKey,
  };

  var body = JSON.stringify({
    model: config.modelId,
    messages: msgs,
    tools: [IDA_SCRIPT_TOOL],
    tool_choice: "auto",
  });

  var headersJson = JSON.stringify(headers);
  var responseStr = fetchFn(url, "POST", headersJson, body);
  return JSON.parse(responseStr);
}

/**
 * Run one turn of the agent loop.
 */
function runTurn(turnNum) {
  eventLog.push({ type: "turn_start", turn: turnNum });
  eventLog.push({ type: "thinking" });

  var apiMessages = [{ role: "system", content: config.systemPrompt }];
  for (var i = 0; i < messages.length; i++) {
    apiMessages.push(messages[i]);
  }

  var response = callLLM(apiMessages);

  if (response.error) {
    eventLog.push({ type: "error", message: response.error.message || JSON.stringify(response.error) });
    return null;
  }

  var choice = response.choices && response.choices[0];
  if (!choice) {
    eventLog.push({ type: "error", message: "No choices in LLM response" });
    return null;
  }

  var assistantMsg = choice.message;
  messages.push(assistantMsg);

  if (assistantMsg.content) {
    eventLog.push({ type: "text", text: assistantMsg.content });
  }

  eventLog.push({ type: "turn_end", turn: turnNum });
  return assistantMsg;
}

/**
 * Execute tool calls from an assistant message.
 */
function executeToolCalls(assistantMsg) {
  var toolCalls = assistantMsg.tool_calls;
  if (!toolCalls || toolCalls.length === 0) return false;

  for (var i = 0; i < toolCalls.length; i++) {
    var tc = toolCalls[i];
    var args;
    try {
      args = typeof tc["function"].arguments === "string"
        ? JSON.parse(tc["function"].arguments)
        : tc["function"].arguments;
    } catch (e) {
      args = { code: tc["function"].arguments };
    }

    eventLog.push({
      type: "tool_call",
      toolCallId: tc.id,
      toolName: tc["function"].name,
      code: args.code || "",
    });

    var result;
    if (tc["function"].name === "ida_script" && toolCallback) {
      try {
        result = toolCallback(args.code || "");
        if (result == null) result = "(no output)";
        result = String(result);
      } catch (e) {
        result = "Script error: " + (e.message || String(e));
      }
    } else {
      result = "Unknown tool: " + tc["function"].name;
    }

    eventLog.push({
      type: "tool_result",
      toolCallId: tc.id,
      toolName: tc["function"].name,
      output: result,
      isError: false,
    });

    messages.push({
      role: "tool",
      tool_call_id: tc.id,
      content: result,
    });
  }

  return true;
}

/**
 * Send a prompt and run the agent loop until completion.
 */
function prompt(message, maxTurns) {
  if (!config) {
    return { events: [], error: "Agent not initialized. Call init() first." };
  }

  maxTurns = maxTurns || 20;
  eventLog = [];

  messages.push({ role: "user", content: message });

  var turnCount = 0;

  while (turnCount < maxTurns) {
    turnCount++;

    var assistantMsg;
    try {
      assistantMsg = runTurn(turnCount);
    } catch (e) {
      eventLog.push({ type: "error", message: "LLM call failed: " + (e.message || String(e)) });
      return { events: eventLog, error: e.message || String(e) };
    }

    if (!assistantMsg) break;

    var hadToolCalls;
    try {
      hadToolCalls = executeToolCalls(assistantMsg);
    } catch (e) {
      eventLog.push({ type: "error", message: "Tool execution failed: " + (e.message || String(e)) });
      return { events: eventLog, error: e.message || String(e) };
    }

    if (!hadToolCalls) break;
  }

  eventLog.push({ type: "done", turns: turnCount });
  return { events: eventLog };
}

/**
 * Reset conversation history.
 */
function reset() {
  messages = [];
  eventLog = [];
}

/**
 * Get current messages.
 */
function getMessages() {
  return messages;
}

// CJS export
if (typeof module !== "undefined") {
  module.exports = { init: init, prompt: prompt, reset: reset, getMessages: getMessages, PROVIDERS: PROVIDERS };
}
