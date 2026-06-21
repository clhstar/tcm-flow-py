# DeerFlow Streaming Parity Failure Note

## Goal

Verify whether tcm-flow can implement token-level typewriter output using DeerFlow's direct pattern:

```python
agent = create_agent(...)
agent.astream(..., stream_mode=["messages", "values"])
```

## Observed Result

The parity test `tests.test_deerflow_create_agent_streaming` did not emit multiple `AIMessageChunk` payloads.

In both the current environment and the DeerFlow-constrained environment, the `messages` stream produced one full `AIMessage` payload for the fake model response.

## Installed Versions

Current `.venv` baseline:

```text
langchain==1.3.4
langgraph==1.2.4
langchain-core==1.4.2
langchain-openai==1.2.2
langgraph-prebuilt==1.1.0
langgraph-checkpoint==4.1.1
langgraph-sdk==0.4.2
```

DeerFlow-constrained `.venv-deerflow-streaming`:

```text
langchain==1.2.15
langgraph==1.1.9
langchain-core==1.3.3
langchain-openai==1.2.1
langgraph-prebuilt==1.0.11
langgraph-checkpoint==4.0.2
langgraph-sdk==0.3.13
```

The isolated install also resolved `langgraph-checkpoint-postgres==3.0.5` under the constraints.

## Event Sequence

The DeerFlow-constrained diagnostic run printed:

```text
[('AIMessage', 'ai', 'abc', {'ls_integration': 'langchain_create_agent', 'langgraph_step': 1, 'langgraph_node': 'model', 'langgraph_triggers': ('branch:to:model',), 'langgraph_path': ('__pregel_pull', 'model'), 'langgraph_checkpoint_ns': 'model:070f0d92-7f80-6cf7-03d7-9dafedc160cc'})]
```

The parity test assertion observed:

```text
[('AIMessage', 'abc', 'model')]
```

## Related Verification

The gateway streaming subset still passed in the DeerFlow-constrained environment:

```text
python -m unittest tests.gateway.test_threads_router tests.gateway.test_thread_run_services
Ran 7 tests in 0.164s
OK
```

## Decision

Do not implement synthetic chunking. The next investigation must compare model provider behavior and middleware behavior against DeerFlow before changing production streaming code.
