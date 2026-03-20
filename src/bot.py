import discord
from discord.ext import commands
import asyncio
import os
from rag_engine import RAGEngine
import config

# Initialize components
intents = discord.Intents.default()
intents.message_content = True
intents.dm_messages = True

bot = commands.Bot(command_prefix='!', intents=intents)
rag_engine = None

@bot.event
async def on_ready():
    global rag_engine
    print(f'Logged in as {bot.user} (ID: {bot.user.id})')
    print('Initializing RAG Engine...')
    try:
        rag_engine = RAGEngine()
        print('RAG Engine initialized successfully.')
    except Exception as e:
        print(f'Failed to initialize RAG Engine: {e}')

@bot.command(name='ingest')
@commands.is_owner()
async def ingest(ctx):
    """
    (Owner) Triggers ingestion.
    - If attachments are present, ingests them.
    - If text follows the command, ingests it as a new rule.
    - If neither, ingests all files in the data directory.
    """
    if not rag_engine:
        await ctx.send("RAG Engine not initialized.")
        return

    import time # Local import to avoid global changes if strictly handling here

    files_to_process = []
    
    # 1. Handle Attachments
    for attachment in ctx.message.attachments:
        if attachment.filename.endswith('.pdf') or attachment.filename.endswith('.txt'):
            file_path = os.path.join(config.DATA_DIR, attachment.filename)
            await attachment.save(file_path)
            files_to_process.append(file_path)
            await ctx.send(f"Downloaded attachment: {attachment.filename}")
        else:
            await ctx.send(f"Skipping unsupported attachment: {attachment.filename}")

    # 2. Handle Text Content
    # Strip the command "!ingest" and whitespace
    content = ctx.message.content.replace(ctx.prefix + ctx.command.name, '', 1).strip()
    if content:
        # Create a new text file for this rule
        filename = f"rule_{int(time.time())}.txt"
        file_path = os.path.join(config.DATA_DIR, filename)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        files_to_process.append(file_path)
        await ctx.send(f"Saved new rule as: {filename}")

    # 3. Execution
    if files_to_process:
        await ctx.send(f"Ingesting {len(files_to_process)} specific item(s)...")
        loop = asyncio.get_running_loop()
        try:
            # Process strictly the new files
            count = 0
            for file_path in files_to_process:
                await loop.run_in_executor(None, rag_engine.ingest_data, file_path)
                count += 1
            await ctx.send(f"Successfully ingested {count} item(s).")
        except Exception as e:
            print(f"Error during partial ingestion: {e}")
            await ctx.send("There was an error performing that command.")
    else:
        # Fallback: scan entire directory
        await ctx.send("Starting full directory ingestion process... check console for details.")
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, rag_engine.ingest_data, config.DATA_DIR)
            await ctx.send("Full ingestion complete!")
        except Exception as e:
            print(f"Error during ingestion: {e}")
            await ctx.send("There was an error performing that command.")

@bot.command(name='clear_db')
@commands.is_owner()
async def clear_db(ctx):
    """(Owner) Clears the vector database."""
    if not rag_engine:
        await ctx.send("RAG Engine not initialized.")
        return
        
    await ctx.send("Clearing database...")
    try:
        rag_engine.clear_database()
        await ctx.send("Database cleared.")
    except Exception as e:
        print(f"Error clearing database: {e}")
        await ctx.send("There was an error performing that command.")

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    # Process commands first
    await bot.process_commands(message)
    
    # If not a command and mentioned or in DM, Query RAG
    is_direct_message = isinstance(message.channel, discord.DMChannel)
    is_mentioned = bot.user.mentioned_in(message)

    if (is_direct_message or is_mentioned) and not message.content.startswith(bot.command_prefix):
        if not rag_engine:
            await message.channel.send("System is starting up or RAG engine is down.")
            return

        async with message.channel.typing():
            # Clean content (remove mention)
            query_text = message.content.replace(f'<@{bot.user.id}>', '').strip()
            
            # Simple "thinking" log
            print(f"Processing query from {message.author}: {query_text}")
            
            # Run query in executor
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(None, rag_engine.query, query_text)
            
            await message.channel.send(response)

def main():
    if not config.DISCORD_TOKEN:
        print("Error: DISCORD_TOKEN not found in environment variables.")
        return
    
    bot.run(config.DISCORD_TOKEN)

if __name__ == '__main__':
    main()
