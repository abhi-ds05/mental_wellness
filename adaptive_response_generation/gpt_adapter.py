import os
import sys

try:
    from openai import OpenAI
except ImportError:
    raise ImportError(
        "You need to install the OpenAI Python SDK:\n\npip install openai"
    )

# ========= CONFIG =========
# Set your OpenAI API key as an environment variable for security:
# Windows CMD:   setx OPENAI_API_KEY "your_api_key"
# Linux/Mac:     export OPENAI_API_KEY="your_api_key"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set.")

# GPT model to use
DEFAULT_MODEL = "gpt-4o-mini"  # You can change to gpt-4o, gpt-4-turbo, gpt-3.5-turbo

# Initialize the client
client = OpenAI(api_key=OPENAI_API_KEY)


def generate_gpt_response(prompt, system_message=None, model=DEFAULT_MODEL, max_tokens=400, temperature=0.7):
    """
    Generates a GPT response for a given prompt.
    Args:
        prompt (str): The user/content prompt to send.
        system_message (str): Optional system instruction to guide GPT output.
        model (str): The GPT model to use.
        max_tokens (int): Max tokens to generate.
        temperature (float): Creativity/variance in output.

    Returns:
        str: The AI-generated text, or None if error.
    """
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": prompt})

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        # Correct attribute access for message content
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"[ERROR] GPT API call failed: {e}")
        return None


def generate_empathetic_message(user_context, strategies=None, mindfulness=None):
    """
    Generates a natural empathetic message given structured context.

    Args:
        user_context (dict): Should contain 'user_id', 'top_emotion', 'mood_trend', 'tone_category', 'recent_journal'
        strategies (list): Optional list of coping strategy suggestions.
        mindfulness (list): Optional list of mindfulness suggestions.

    Returns:
        str: AI-generated empathetic support text.
    """
    system_msg = (
        "You are a warm, empathetic mental wellness assistant. "
        "Respond in a friendly, supportive tone. Use encouraging language, avoid clinical jargon, "
        "and provide compassionate responses aimed at emotional support, not diagnosis."
    )

    # Build dynamic prompt from context
    prompt_parts = [
        f"User ID: {user_context.get('user_id')}",
        f"Top Emotion Detected: {user_context.get('top_emotion')}",
        f"Mood Trend: {user_context.get('mood_trend')}",
        f"Tone Category: {user_context.get('tone_category')}"
    ]

    if user_context.get("recent_journal"):
        prompt_parts.append(f"Recent Journal Entry Snippet: {user_context['recent_journal']}")

    if strategies:
        prompt_parts.append(f"Suggested Coping Strategies: {', '.join(strategies)}")

    if mindfulness:
        prompt_parts.append(f"Suggested Mindfulness Exercises: {', '.join(mindfulness)}")

    prompt_parts.append(
        "\nGenerate an empathetic, supportive, and encouraging paragraph for the user, "
        "referencing their emotional state and including at least one suggested action they could take next."
    )

    prompt_str = "\n".join(prompt_parts)

    return generate_gpt_response(prompt_str, system_message=system_msg)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test GPT Adapter for Empathetic Responses")
    parser.add_argument("prompt_or_userid", type=str, help="Custom prompt text or user_id for context demo")
    parser.add_argument("--demo", action="store_true", help="Run a demo with fake user context instead of raw prompt")
    args = parser.parse_args()

    if args.demo:
        # Demo mode with fake user context
        sample_context = {
            "user_id": args.prompt_or_userid,
            "top_emotion": "sadness",
            "mood_trend": "declining",
            "tone_category": "negative",
            "recent_journal": "I've been feeling really low energy lately and haven't wanted to talk to anyone."
        }
        strategies_demo = [
            "Call or message a supportive friend",
            "Step outside for a short walk in nature"
        ]
        mindfulness_demo = [
            "Try a 5-minute mindful breathing exercise",
            "Do a gentle body scan meditation"
        ]
        reply = generate_empathetic_message(
            sample_context,
            strategies=strategies_demo,
            mindfulness=mindfulness_demo
        )
        print("\n[GPT Empathetic Reply]\n", reply)
    else:
        # Raw prompt mode
        reply = generate_gpt_response(args.prompt_or_userid)
        print("\n[GPT Direct Reply]\n", reply)
