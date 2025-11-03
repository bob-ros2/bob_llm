#!/bin/bash

PROMPT_TOPIC=${1:-"llm_prompt"}
RESULT_TOPIC=${2:-"llm_response"}

[ $# -gt 2 -o "$1" = "-h" ] \
  && echo "Simple String topic chat" \
  && echo "Usage: $(basename $0) [<input-topic> [<output-topic>]]" \
  && exit

cleanup() {
    echo -e "\n\nShutting down listener and exiting..."
    if kill -0 "$LISTENER_PID" 2>/dev/null; then
        kill "$LISTENER_PID"
    fi
    exit 0
}

trap cleanup SIGINT

# Start listening to the result topic in the background
echo "--- Listening for results on $RESULT_TOPIC ---"
ros2 topic echo -f "$RESULT_TOPIC" std_msgs/msg/String &
LISTENER_PID=$!

echo "--- Enter your prompt below (Press Ctrl+C to exit) ---"
while true; do
    # Prompt the user for input
    read -p "> " prompt

    if [[ -z "$prompt" ]]; then
        continue
    fi

    ros2 topic pub --keep-alive 1.0 --once "$PROMPT_TOPIC" std_msgs/msg/String "data: '$prompt'"
done