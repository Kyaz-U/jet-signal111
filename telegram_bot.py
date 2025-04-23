import logging
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
from predictor import predict_aviator_signal

TOKEN = "8042150137:AAHDAYzFd8hqe2nvckl21slLak_pg8HPEYo"
USER_ID = 92058415

logging.basicConfig(level=logging.INFO)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id == USER_ID:
        signal = predict_aviator_signal()
        await context.bot.send_message(chat_id=update.effective_chat.id, text=signal, parse_mode="HTML")
    else:
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Sizga ruxsat yo'q.")

def start_bot():
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("signal", start))
    app.run_polling()