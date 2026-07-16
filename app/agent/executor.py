# Phase 3: Async task executor (MQ-driven)
# Queues:
#   agent_task_queue     — long-running agent jobs
#   document_parse_queue — PRD / API doc ingestion into RAG
#   tool_execution_queue — GitHub / DB tool calls
#
# Will integrate with RabbitMQ (or Redis Streams) in Phase 3.
