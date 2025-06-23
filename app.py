import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import Counter
from wordcloud import WordCloud
import emoji
import nltk
from nltk.corpus import stopwords
from fpdf import FPDF
import base64
from statsmodels.tsa.arima.model import ARIMA
import random
from textblob import TextBlob

nltk.download('stopwords')
import bcrypt
import json
import os

CREDENTIALS_FILE = "users.json"

def load_users():
    if os.path.exists(CREDENTIALS_FILE):
        with open(CREDENTIALS_FILE, "r") as file:
            return json.load(file)
    return {}

def save_users(users):
    with open(CREDENTIALS_FILE, "w") as file:
        json.dump(users, file)

def register_user(username, password):
    users = load_users()
    if username in users:
        return False
    hashed_pw = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    users[username] = hashed_pw
    save_users(users)
    return True

def authenticate_user(username, password):
    users = load_users()
    if username in users and bcrypt.checkpw(password.encode(), users[username].encode()):
        return True
    return False


# ===================== Preprocessing Function =====================
def preprocess_chat(data):
    data = data.replace('\u202f', ' ').replace('\u200e', '')
    pattern = r'(\d{1,2}/\d{1,2}/\d{4}), (\d{1,2}:\d{2}) (am|pm) - (.*?): (.*)'
    matches = re.findall(pattern, data, flags=re.IGNORECASE)

    records = []
    for date, time, ampm, user, message in matches:
        timestamp = f"{date} {time} {ampm}"
        try:
            dt = pd.to_datetime(timestamp, format='%d/%m/%Y %I:%M %p')
            if message.strip().lower() not in ["<media omitted>", "null"]:
                records.append((dt, user.strip(), message.strip()))
        except:
            continue

    return pd.DataFrame(records, columns=['datetime', 'user', 'message'])

# ===================== Analysis Functions =====================
def filter_data(df, selected_users, start_date, end_date):
    if selected_users:
        df = df[df['user'].isin(selected_users)]
    if start_date and end_date:
        df = df[(df['datetime'] >= pd.to_datetime(start_date)) & (df['datetime'] <= pd.to_datetime(end_date))]
    return df

def weekly_activity(df):
    df['weekday'] = df['datetime'].dt.day_name()
    return df['weekday'].value_counts().sort_index()

def monthly_activity(df):
    df['month'] = df['datetime'].dt.strftime('%B %Y')
    return df['month'].value_counts().sort_index()

def hourly_distribution(df):
    df['hour'] = df['datetime'].dt.hour
    return df['hour'].value_counts().sort_index()

def heatmap_data(df):
    df['hour'] = df['datetime'].dt.hour
    df['day'] = df['datetime'].dt.day_name()
    return df.pivot_table(index='day', columns='hour', values='message', aggfunc='count').fillna(0)

def common_words(df):
    stop_words = set(stopwords.words('english'))
    words = ' '.join(df['message']).lower()
    words = re.findall(r'\b\w+\b', words)
    filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
    return Counter(filtered_words).most_common(20)

def user_frequency(df):
    return df['user'].value_counts()

def generate_wordcloud(df):
    text = ' '.join(df['message'])
    wc = WordCloud(width=800, height=400, background_color='white').generate(text)
    return wc

def emoji_counter(df):
    emojis = []
    for msg in df['message']:
        emojis += [c for c in msg if c in emoji.EMOJI_DATA]
    return Counter(emojis).most_common(10)

def user_message_stats(df):
    df['length'] = df['message'].apply(len)
    avg_lengths = df.groupby('user')['length'].mean().sort_values(ascending=False)
    daily_counts = df.groupby(['user', df['datetime'].dt.date]).size().groupby('user').mean()
    return avg_lengths, daily_counts

def sentiment_analysis(df):
    df['polarity'] = df['message'].apply(lambda msg: TextBlob(msg).sentiment.polarity)
    user_sentiment = df.groupby('user')['polarity'].mean().sort_values()
    return user_sentiment

def predict_busiest_month(df, steps=3):
    df['month'] = df['datetime'].dt.to_period('M')
    monthly_counts = df.groupby('month').size()
    monthly_counts.index = monthly_counts.index.to_timestamp()
    if len(monthly_counts) < 3:
        return None, None, "Not enough data for a trend."
    model = ARIMA(monthly_counts, order=(1, 1, 1))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)
    predicted_month = forecast.idxmax()
    explanation = (
        f"The prediction is based on recent trends. "
        f"The forecast shows the highest expected activity in {predicted_month.strftime('%B %Y')} "
        f"with an estimated {int(forecast.max())} messages."
    )
    return predicted_month.strftime('%B %Y'), forecast, explanation

def generate_future_messages(df, num_sentences=5):
    text = ' '.join(df['message'].dropna())
    words = text.split()
    if len(words) < 2:
        return ["Not enough data to predict future messages."]
    markov_chain = {}
    for i in range(len(words) - 1):
        key, next_word = words[i], words[i + 1]
        markov_chain.setdefault(key, []).append(next_word)
    messages = []
    for _ in range(num_sentences):
        word = random.choice(words)
        sentence = [word]
        for _ in range(random.randint(4, 12)):
            word = random.choice(markov_chain.get(word, words))
            sentence.append(word)
        messages.append(' '.join(sentence).capitalize() + '.')
    return messages

def predict_future_words(df, num_words=10):
    text = ' '.join(df['message'].dropna()).lower()
    words = re.findall(r'\b\w+\b', text)
    if len(words) < 2:
        return ["Not enough data to predict future words."]

    markov_chain = {}
    for i in range(len(words) - 1):
        key = words[i]
        next_word = words[i + 1]
        markov_chain.setdefault(key, []).append(next_word)

    current_word = random.choice(words)
    future_words = [current_word]
    for _ in range(num_words - 1):
        next_words = markov_chain.get(current_word, words)
        current_word = random.choice(next_words)
        future_words.append(current_word)

    return future_words

def predict_active_days(df):
    weekday_counts = df['datetime'].dt.day_name().value_counts()
    top_days = weekday_counts.head(2).index.tolist()
    return top_days, f"Based on past data, most messages are usually sent on {', '.join(top_days)}."

def predict_active_hours(df):
    hour_counts = df['datetime'].dt.hour.value_counts().sort_index()
    peak_hours = hour_counts[hour_counts == hour_counts.max()].index.tolist()
    return peak_hours, f"Most activity has been observed around {', '.join(str(h) + ':00' for h in peak_hours)}."

def detect_message_trend(df):
    df['month'] = df['datetime'].dt.to_period('M')
    monthly = df.groupby('month').size()
    if len(monthly) < 3:
        return "Not enough data to detect trend."
    trend = "increasing" if monthly.iloc[-1] > monthly.iloc[0] else "decreasing"
    return f"Message volume is generally {trend} over time."

# ===================== Streamlit App =====================
st.set_page_config(layout='wide')
st.title("🔐 WhatsApp Chat Analyzer - Login Required")

menu = ["Login", "Register"]
choice = st.sidebar.selectbox("Menu", menu)

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if choice == "Login":
    st.subheader("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if authenticate_user(username, password):
            st.session_state.authenticated = True
            st.success(f"Welcome {username}!")
        else:
            st.error("Invalid credentials")

elif choice == "Register":
    st.subheader("Create New Account")
    new_user = st.text_input("New Username")
    new_password = st.text_input("New Password", type="password")
    if st.button("Register"):
        if register_user(new_user, new_password):
            st.success("Registration successful. Please log in.")
        else:
            st.error("Username already exists.")

if not st.session_state.authenticated:
    st.stop()
# ========== Smart WhatsApp Tools Page ==========
st.sidebar.markdown("---")
if st.sidebar.button("💬 Launch Smart WhatsApp Tools"):
    st.session_state.page = "smart_tools"

if st.session_state.get("page") == "smart_tools":
    st.title("🚀 Smart WhatsApp Tools")

    st.markdown("### 💬 AI-Assisted Messaging")
    user_input = st.text_input("Start typing your message...")
    suggestions = []
    if user_input:
        keywords = ["party", "hello", "meeting", "fun", "food"]
        ai_responses = {
            "party": ["Let’s party tonight!", "Who's bringing snacks?", "Can we reschedule the party?"],
            "hello": ["Hello there!", "Hey! How's it going?", "Hi, long time!"],
            "meeting": ["Can we postpone the meeting?", "Meeting in 10 minutes!", "Zoom link please!"],
        }
        for word in keywords:
            if word in user_input.lower():
                suggestions = ai_responses.get(word, [])
                break

    if suggestions:
        st.markdown("**Suggested Messages:**")
        for s in suggestions:
            st.write(f"👉 {s}")

    st.markdown("---")
    st.markdown("### ⏲️ Timer Message")
    future_msg = st.text_area("Your message")
    delay = st.slider("Send after (seconds)", min_value=5, max_value=300, step=5)
    if st.button("🕒 Schedule Message"):
        st.success(f"Message will be 'sent' in {delay} seconds (simulation).")
        st.code(f"> {future_msg}")

    st.markdown("---")
    st.markdown("### 🎭 Voice Message Changer (Experimental)")
    voice = st.file_uploader("Upload a voice message (.wav)", type=["wav"])
    if voice:
        voice_type = st.selectbox("Choose new voice", ["Robot", "Male", "Female", "Alien"])
        st.audio(voice, format="audio/wav")
        if st.button("Transform Voice"):
            st.info(f"Voice transformed to {voice_type}. (placeholder - needs TTS/Voice SDK)")

    st.markdown("---")
    st.markdown("### 🛠️ Coming Soon:")
    st.markdown("- Auto responder\n- Smart cleanup\n- Chat mood analytics\n- Multi-chat linking")
    st.stop()


uploaded_file = st.file_uploader("📂 Upload your WhatsApp chat (.txt file)", type="txt")

if uploaded_file is not None:
    raw_text = uploaded_file.read().decode('utf-8')
    df = preprocess_chat(raw_text)

    if not df.empty:
        st.sidebar.header("🧮 Choose Analysis")
        all_users = df['user'].unique().tolist()
        selected_users = st.sidebar.multiselect("👤 Filter by User(s)", all_users)
        start_date = st.sidebar.date_input("📅 Start Date", value=df['datetime'].min().date())
        end_date = st.sidebar.date_input("📅 End Date", value=df['datetime'].max().date())
        df = filter_data(df, selected_users, start_date, end_date)

        if st.sidebar.checkbox("📄 Show Raw Data"):
            st.subheader("Raw Chat Data (Filtered)")
            st.dataframe(df)

        if st.sidebar.checkbox("📊 User-Specific Stats"):
            st.subheader("📈 User Message Stats")
            avg_lengths, avg_msgs_per_day = user_message_stats(df)
            st.write("**📏 Avg. Message Length by User:**")
            st.dataframe(avg_lengths.rename("Avg Length"))
            st.write("**📆 Avg. Messages per Day by User:**")
            st.dataframe(avg_msgs_per_day.rename("Avg Msgs/Day"))

        if st.sidebar.checkbox("🎭 Sentiment Analysis"):
            st.subheader("🎯 Sentiment Analysis by User")
            sentiments = sentiment_analysis(df)
            df['polarity'] = df['message'].apply(lambda msg: TextBlob(msg).sentiment.polarity)
            st.write("**User Sentiment Polarity (−1 = Negative, +1 = Positive):**")
            st.dataframe(sentiments.rename("Polarity"))
            most_pos = sentiments.idxmax()
            most_neg = sentiments.idxmin()

            st.success(f"😊 Most Positive User: {most_pos} ({sentiments.max():.2f})")
            st.error(f"😠 Most Negative User: {most_neg} ({sentiments.min():.2f})")

            st.subheader(f"🔍 Messages Contributing to Negative Sentiment for {most_neg}")
            neg_msgs = df[(df['user'] == most_neg) & (df['polarity'] < 0)].sort_values(by='polarity')
            if neg_msgs.empty:
                st.info("No clearly negative messages were found for this user.")
            else:
                st.markdown(f"These are the messages from **{most_neg}** with negative sentiment scores, which contributed to their overall polarity of **{sentiments[most_neg]:.2f}**.")
                st.dataframe(neg_msgs[['datetime', 'message', 'polarity']].reset_index(drop=True))


            st.subheader(f"✨ Positive Messages from {most_pos}")
            pos_msgs = df[(df['user'] == most_pos) & (df['polarity'] > 0)].sort_values(by='polarity', ascending=False)
            if pos_msgs.empty:
                st.info("No clearly positive messages were found for this user.")
            else:
                st.dataframe(pos_msgs[['datetime', 'message', 'polarity']].reset_index(drop=True))

        if st.sidebar.checkbox("📊 Multi-User Comparison"):
            st.subheader("🔄 Multi-User Message Comparison")
            user_counts = df['user'].value_counts()
            fig, ax = plt.subplots()
            sns.barplot(x=user_counts.index, y=user_counts.values, ax=ax, palette="coolwarm")
            ax.set_title("Messages per User")
            ax.set_ylabel("Messages")
            ax.set_xticklabels(user_counts.index, rotation=45, ha='right')
            st.pyplot(fig)

        if st.sidebar.checkbox("📅 Weekly Activity"):
            st.subheader("🗓️ Weekly Chat Activity")
            weekly = weekly_activity(df)
            fig, ax = plt.subplots()
            sns.barplot(x=weekly.index, y=weekly.values, palette='viridis', ax=ax)
            ax.set_ylabel("Message Count")
            ax.set_xlabel("Day")
            st.pyplot(fig)

        if st.sidebar.checkbox("📆 Monthly Activity"):
            st.subheader("📈 Monthly Chat Activity")
            monthly = monthly_activity(df)
            fig, ax = plt.subplots()
            monthly.plot(kind='bar', color='skyblue', ax=ax)
            ax.set_ylabel("Messages")
            ax.set_xlabel("Month")
            st.pyplot(fig)

        if st.sidebar.checkbox("🕒 Hourly Activity"):
            st.subheader("⌛ Most Active Hours")
            hourly = hourly_distribution(df)
            fig, ax = plt.subplots()
            sns.lineplot(x=hourly.index, y=hourly.values, marker='o', ax=ax)
            ax.set_xticks(range(0, 24))
            ax.set_xlabel("Hour of Day")
            ax.set_ylabel("Message Count")
            st.pyplot(fig)

        if st.sidebar.checkbox("📊 Activity Heatmap"):
            st.subheader("🔥 Activity Heatmap")
            pivot = heatmap_data(df)
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.heatmap(pivot, cmap="YlGnBu", ax=ax, linewidths=0.5)
            ax.set_title("Messages per Hour by Day")
            st.pyplot(fig)

        if st.sidebar.checkbox("🗣️ Most Common Words"):
            st.subheader("💬 Most Common Words")
            common = common_words(df)
            common_df = pd.DataFrame(common, columns=['Word', 'Frequency'])
            st.dataframe(common_df)

        if st.sidebar.checkbox("☁️ Word Cloud"):
            st.subheader("☁️ Word Cloud")
            wc = generate_wordcloud(df)
            fig, ax = plt.subplots()
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)

        if st.sidebar.checkbox("🙋 Chat Frequency by User"):
            st.subheader("📊 Chat Frequency by User")
            freq = user_frequency(df)
            fig, ax = plt.subplots()
            sns.barplot(x=freq.index, y=freq.values, palette='Set2', ax=ax)
            ax.set_xticklabels(freq.index, rotation=45, ha='right')
            ax.set_ylabel("Messages")
            st.pyplot(fig)

        if st.sidebar.checkbox("😂 Emoji Usage"):
            st.subheader("😂 Top Emojis Used")
            emojis = emoji_counter(df)
            emoji_df = pd.DataFrame(emojis, columns=['Emoji', 'Count'])
            st.dataframe(emoji_df)

        if st.sidebar.checkbox("🔮 Predict Busiest Month"):
            st.subheader("📅 Predicted Busiest Upcoming Month")
            month, forecast, reason = predict_busiest_month(df)
            if month:
                st.success(f"🚀 Predicted busiest month is: **{month}**")
                st.write(reason)
                st.line_chart(forecast)
            else:
                st.warning(reason)

        if st.sidebar.checkbox("💬 Predict Future Messages"):
            st.subheader("🗯️ AI-Predicted Future Messages")
            predictions = generate_future_messages(df)
            for msg in predictions:
                st.write(f"• {msg}")
        
        if st.sidebar.checkbox("💡 Predict Future Words"):
            st.subheader("🔤 Predicted Future Words")
            predicted_words = predict_future_words(df)
            st.write(" ".join(predicted_words))

        if st.sidebar.checkbox("🗓️ Predict Most Active Days"):
            st.subheader("🗓️ Likely Active Weekdays")
            days, reason = predict_active_days(df)
            st.info(reason)

        if st.sidebar.checkbox("⏳ Predict Most Active Hours"):
            st.subheader("⏰ Likely Active Hours")
            hours, reason = predict_active_hours(df)
            st.info(reason)

        if st.sidebar.checkbox("📈 Message Volume Trend"):
            st.subheader("📉 Message Trend Analysis")
            st.info(detect_message_trend(df))

    else:
        st.warning("⚠️ No valid messages found to analyze. Please try another file.")
