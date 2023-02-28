import slack
import os
from pathlib import Path
from dotenv import load_dotenv
from flask import Flask
from slackeventsapi import SlackEventAdapter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.cohere import CohereEmbeddings
from langchain.text_splitter import NLTKTextSplitter
from langchain.vectorstores.elastic_vector_search import ElasticVectorSearch
from langchain.document_loaders import PagedPDFSplitter
from langchain.vectorstores import FAISS
from langchain.chains import VectorDBQAWithSourcesChain
from langchain import OpenAI
from datetime import datetime, timedelta
import time

env_path = Path('.') / '.env'

load_dotenv(dotenv_path=env_path)

app = Flask(__name__)

slack_event_adapter = SlackEventAdapter(os.environ['SIGNING_SECRET'], "/slack/events", app)

client = slack.WebClient(token=os.environ['SLACK_TOKEN'])

BOT_ID = client.api_call("auth.test")['user_id']

welcome_messages = {}


def configure_qa_chain():
    devsu_loader = PagedPDFSplitter("./files/Devsu Employee Handbook.pdf")
    devsu_pages = devsu_loader.load_and_split(NLTKTextSplitter(chunk_size=2000, chunk_overlap=100))
    embeddings = OpenAIEmbeddings()
    faiss_index = FAISS.from_documents(devsu_pages, OpenAIEmbeddings())
    return VectorDBQAWithSourcesChain.from_chain_type(OpenAI(temperature=0), chain_type="map_rerank",
                                                       vectorstore=faiss_index)


chain = configure_qa_chain()


def get_response_from_book(text):
    return chain({"question": text})["answer"]


class WelcomeMessage:
    START_TEXT = {
        'type': 'section',
        'text': {
            'type': 'mrkdwn',
            'text': (
                'Welcome to the Devsu\'s handbook! \n\n'
                '*You can start asking questions about the handbook*'
            )
        }
    }

    DIVIDER = {'type': 'divider'}

    def __init__(self, channel):
        self.channel = channel
        self.icon_emoji = ':robot_face:'
        self.timestamp = ''
        self.completed = False

    def get_message(self):
        return {
            'ts': self.timestamp,
            'channel': self.channel,
            'username': 'Welcome Robot!',
            'icon_emoji': self.icon_emoji,
            'blocks': [
                self.START_TEXT,
                self.DIVIDER,
            ]
        }


def send_welcome_message(channel, user):
    if channel not in welcome_messages:
        welcome_messages[channel] = {}

    if user in welcome_messages[channel]:
        return

    welcome = WelcomeMessage(channel)
    message = welcome.get_message()
    response = client.chat_postMessage(**message)
    welcome.timestamp = response['ts']

    welcome_messages[channel][user] = welcome


@slack_event_adapter.on('message')
def message(payload):
    event = payload.get('event', {})
    channel_id = event.get('channel')
    user_id = event.get('user')
    text = event.get('text')
    if user_id is not None and BOT_ID is not user_id:
        if text.lower() == "start":
            send_welcome_message(f"@{user_id}", user_id)
        else:
            response = get_response_from_book(text)
            client.chat_postMessage(channel=f"@{user_id}", text=f"{response}")


if __name__ == "__main__":
    app.run(debug=True)
