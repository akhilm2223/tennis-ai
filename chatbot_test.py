#!/usr/bin/env python3
"""
Interactive CLI Chatbot - Chat only
"""

import sys
import textwrap
import dotenv
from MentalCoach.coach.coach import MentalCoachChatbot

# Load env vars
dotenv.load_dotenv()

# Terminal wrapper (for readable console text)
WRAP = 88
def wrap(text):
    return "\n".join(textwrap.wrap(text, WRAP))


def print_header():
    print("=" * WRAP)
    print("🏆  MENTAL COACH — Chat".center(WRAP))
    print("=" * WRAP)
    print("Ask about mental toughness, confidence, performance, pressure, focus, etc.")
    print("\nCommands:")
    print("  help            Show commands")
    print("  quit            Exit")
    print("=" * WRAP)
    print()


def main():
    print_header()

    # Init chatbot
    try:
        print("🔧 Initializing chatbot...")
        bot = MentalCoachChatbot()
        print("✅ Chatbot ready!\n")
    except Exception as e:
        print(f"❌ Error loading chatbot: {e}")
        return 1

    # Use a consistent session ID for this test session
    session_id = "test_session"

    # ====================
    #   MAIN LOOP
    # ====================
    while True:
        try:
            user = input("You: ").strip()

            # ---- Commands ----
            if user.lower() in ("quit", "exit", "q"):
                print("\n👋 Goodbye — keep training the mental game!")
                break

            if user.lower() == "help":
                print_header()
                continue

            # ---- Chat ----
            print("💭 Thinking...")
            print("🔍 Searching knowledge base...")
            context = bot.search_pinecone(user)
            if not context:
                print("⚠️  No relevant sources found, but will try to respond anyway")
            
            print("🤖 Generating response...")
            answer = bot.generate_response(user, context, session_id=session_id)

            if not answer:
                print("❌ Sorry, couldn't generate a response.\n")
                continue

            # Print nicely
            print("\n🤖 Coach:\n")
            print(wrap(answer))
            print()

        except KeyboardInterrupt:
            print("\n\n👋 Exiting. Stay strong!")
            break
        except Exception as e:
            print(f"⚠️ Error: {e}\n")
            continue

    return 0


if __name__ == "__main__":
    sys.exit(main())
