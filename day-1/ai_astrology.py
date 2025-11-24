import ollama
from datetime import date
from ollama import Options
from rich.console import Console

def main():
    today = date.today().strftime('%A, %d-%m-%Y')
    #LLM = "qwen3-vl:4b-thinking"
    LLM = "qwen3-vl:4b-instruct"

    # Allow specifying remote Ollama server
    ollama_host = input("Enter Ollama server URL (press Enter for localhost): ").strip()
    if not ollama_host:
        ollama_host = "http://localhost:11434"
    
    # Create Ollama client with specified host
    client = ollama.Client(host=ollama_host)
    
    name = input("Enter your name: ")
    star_sign = input("Enter your star sign: ")

    system_prompt = f"""You are an AI astrology assistant called Maude. Provide a short but interesting, positive and
        optimistic horoscope for tomorrow. Provide the response in Markdown format.
        Remember, the user is looking for a positive and optimistic outlook on their future.
        Use British English, metric and EU date formats where applicable."""

    instruction = f"Please provide a horoscope for {name} who's star sign is {star_sign}. Today's date is {today}."

    response = client.chat( model=LLM, think=True, stream=False,
        messages=[ {'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': instruction} ],
        options=Options( temperature=0.8, num_ctx=4096, top_p=0.95, top_k=40, num_predict=-1 ))

    console = Console()
    
    thinking_started = False
    content_started = False
    
    for chunk in response:
        if hasattr(chunk.message, 'thinking') and chunk.message.thinking:
            if not thinking_started:
                console.print("[bold blue]🤔 Maude's Thinking Process:[/bold blue]")
                thinking_started = True
            console.print(f"[dim]{chunk.message.thinking}[/dim]", end='')
        
        if hasattr(chunk.message, 'content') and chunk.message.content:
            if not content_started:
                if thinking_started:
                    console.print("\n" + "=" * 50 + "\n")
                console.print("[bold magenta]✨ Your Horoscope:[/bold magenta]")
                content_started = True
            console.print(chunk.message.content, end='')
    
    console.print()  # Add a newline at the end

if __name__ == "__main__":
    main()
