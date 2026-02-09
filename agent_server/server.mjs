/**
 * IDA Chat Agent Server
 *
 * Node.js process that runs the pi-mono agent loop and communicates
 * with the Python host via JSON-line protocol over stdin/stdout.
 *
 * Protocol (Python -> Node):
 *   {"type":"init", "systemPrompt":"...", "model":"openrouter/meta-llama/llama-3.3-70b-instruct:free"}
 *   {"type":"prompt", "message":"..."}
 *   {"type":"tool_result", "toolCallId":"...", "content":"...", "isError":false}
 *
 * Protocol (Node -> Python):
 *   {"type":"ready"}
 *   {"type":"thinking"}
 *   {"type":"text", "text":"..."}
 *   {"type":"tool_call", "toolCallId":"...", "toolName":"...", "args":{...}}
 *   {"type":"turn_start", "turn":1}
 *   {"type":"turn_end", "turn":1}
 *   {"type":"done", "turns":2}
 *   {"type":"error", "message":"..."}
 */

import { Agent } from "@mariozechner/pi-agent-core";
import { getModel, Type, registerBuiltInApiProviders } from "@mariozechner/pi-ai";
import { createInterface } from "readline";

// Ensure built-in providers are registered
registerBuiltInApiProviders();

// Use stderr for debug logging (stdout is protocol only)
function log(...args) {
  process.stderr.write(`[agent-server] ${args.join(" ")}\n`);
}

// Send a JSON-line message to Python
function send(msg) {
  process.stdout.write(JSON.stringify(msg) + "\n");
}

// Global state
let agent = null;
let pendingToolResults = new Map(); // toolCallId -> {resolve, reject}
let turnCount = 0;

// Define the ida_script tool schema
const idaScriptParams = Type.Object({
  code: Type.String({ description: "Python code to execute against the IDA database. The code has access to 28 pre-approved analysis functions (enumerate_functions, decompile_function, etc.). Use print() to output results." }),
});

// The ida_script tool bridges back to Python for execution
const idaScriptTool = {
  name: "ida_script",
  label: "IDA Script",
  description: `Execute Python analysis code against an IDA Pro database. The sandbox provides these functions:

Database: get_binary_info()
Functions: enumerate_functions(), get_function_by_name(name), disassemble_function(addr), decompile_function(addr), get_function_signature(addr), get_callers(addr), get_callees(addr), get_basic_blocks(addr)
Cross-refs: get_xrefs_to(addr), get_xrefs_from(addr)
Strings: enumerate_strings(), get_string_at(addr)
Segments: enumerate_segments()
Names: enumerate_names(), get_name_at(addr), demangle_name(name)
Imports/Exports: enumerate_imports(), enumerate_entries()
Memory: read_bytes(addr, size), find_bytes(pattern), get_disassembly_at(addr), get_instruction_at(addr)
Classification: is_code_at(addr), is_data_at(addr), is_valid_address(addr)
Comments: get_comment_at(addr)

Use print() to output results. All functions return JSON-safe types (dicts, lists, ints, strings).`,
  parameters: idaScriptParams,
  execute: async (toolCallId, params, signal, onUpdate) => {
    // Send tool call to Python and wait for result
    send({ type: "tool_call", toolCallId, toolName: "ida_script", args: params });

    // Wait for Python to execute and return result
    const result = await new Promise((resolve, reject) => {
      pendingToolResults.set(toolCallId, { resolve, reject });
    });

    if (result.isError) {
      return {
        content: [{ type: "text", text: `Script error: ${result.content}` }],
        details: { success: false },
      };
    }

    return {
      content: [{ type: "text", text: result.content || "(no output)" }],
      details: { success: true },
    };
  },
};

function handleInit(msg) {
  const [providerName, ...modelParts] = msg.model.split("/");
  const modelId = modelParts.join("/");

  log(`Initializing agent with provider=${providerName}, model=${modelId}`);

  let model;
  try {
    model = getModel(providerName, modelId);
  } catch (e) {
    send({ type: "error", message: `Failed to get model ${msg.model}: ${e.message}` });
    return;
  }

  log(`Model resolved: ${model.name} (api=${model.api}, baseUrl=${model.baseUrl})`);

  agent = new Agent({
    initialState: {
      systemPrompt: msg.systemPrompt || "You are a helpful binary analysis assistant.",
      model,
      thinkingLevel: "off",
      tools: [idaScriptTool],
    },
    getApiKey: async (provider) => {
      // Check environment variables
      const envKeys = {
        openrouter: "OPENROUTER_API_KEY",
        anthropic: "ANTHROPIC_API_KEY",
        openai: "OPENAI_API_KEY",
        google: "GEMINI_API_KEY",
        groq: "GROQ_API_KEY",
      };
      const envVar = envKeys[provider] || `${provider.toUpperCase().replace(/-/g, "_")}_API_KEY`;
      const key = process.env[envVar];
      if (key) {
        log(`Using API key from ${envVar}`);
        return key;
      }
      log(`Warning: No API key found for provider ${provider} (tried ${envVar})`);
      return undefined;
    },
  });

  // Subscribe to agent events for streaming output
  agent.subscribe((event) => {
    switch (event.type) {
      case "turn_start":
        turnCount++;
        send({ type: "turn_start", turn: turnCount });
        send({ type: "thinking" });
        break;

      case "message_update":
        if (event.assistantMessageEvent) {
          const evt = event.assistantMessageEvent;
          if (evt.type === "text_delta") {
            send({ type: "text_delta", text: evt.delta });
          }
        }
        break;

      case "message_end":
        if (event.message && event.message.role === "assistant") {
          // Extract full text from assistant message
          const textParts = [];
          for (const part of event.message.content || []) {
            if (part.type === "text") {
              textParts.push(part.text);
            }
          }
          if (textParts.length > 0) {
            send({ type: "text", text: textParts.join("") });
          }
        }
        break;

      case "turn_end":
        send({ type: "turn_end", turn: turnCount });
        break;

      case "tool_execution_start":
        log(`Tool execution start: ${event.toolName} (${event.toolCallId})`);
        break;

      case "tool_execution_end":
        log(`Tool execution end: ${event.toolName} (${event.toolCallId})`);
        break;

      case "agent_end":
        send({ type: "done", turns: turnCount });
        turnCount = 0;
        break;
    }
  });

  send({ type: "ready" });
  log("Agent initialized and ready");
}

async function handlePrompt(msg) {
  if (!agent) {
    send({ type: "error", message: "Agent not initialized. Send init first." });
    return;
  }

  turnCount = 0;
  log(`Processing prompt: ${msg.message.substring(0, 100)}...`);

  try {
    await agent.prompt(msg.message);
  } catch (e) {
    log(`Agent error: ${e.message}`);
    send({ type: "error", message: e.message });
  }
}

function handleToolResult(msg) {
  const pending = pendingToolResults.get(msg.toolCallId);
  if (pending) {
    pendingToolResults.delete(msg.toolCallId);
    pending.resolve({
      content: msg.content || "",
      isError: msg.isError || false,
    });
  } else {
    log(`Warning: No pending tool call for ${msg.toolCallId}`);
  }
}

// Main: read JSON lines from stdin
const rl = createInterface({ input: process.stdin, terminal: false });

rl.on("line", async (line) => {
  if (!line.trim()) return;

  let msg;
  try {
    msg = JSON.parse(line);
  } catch (e) {
    send({ type: "error", message: `Invalid JSON: ${e.message}` });
    return;
  }

  switch (msg.type) {
    case "init":
      handleInit(msg);
      break;
    case "prompt":
      // Run async, don't await (allows tool_result messages to arrive)
      handlePrompt(msg);
      break;
    case "tool_result":
      handleToolResult(msg);
      break;
    default:
      send({ type: "error", message: `Unknown message type: ${msg.type}` });
  }
});

rl.on("close", () => {
  log("stdin closed, exiting");
  process.exit(0);
});

// Signal ready for connections
log("Agent server started, waiting for init...");
