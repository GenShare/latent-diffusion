#!.venv/bin/python
import json
import json.decoder
import logging
import os
import subprocess
import sys
import time

from os import environ
from threading import Thread

# TODO: add these requirements to pyenv
import boto3
import botocore

from scripts.txt2img import setup, generate

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

def _path_from_uid(uid: str) -> str:
    return f'/tmp/{uid}.png'

RETRY_DELAY = 10

def process_queue(thread_id, s3, sqs, model, sampler):
    while True:
        response = sqs.receive_message(QueueUrl=environ['INBOUND_REQUESTS_QUEUE_URL'],)

        if 'Messages' in response:
            for message in response['Messages']:
                uid, body = message['MessageId'], message['Body']
                sqs.delete_message(
                        QueueUrl=environ['INBOUND_REQUESTS_QUEUE_URL'],
                        ReceiptHandle=message['ReceiptHandle']
                )

                try:
                    body = json.loads(body)
                    prompt = body['prompt']
                except json.decoder.JSONDecodeError:
                    logging.info(f'Unable to parse {uid}: "{body}"')
                    continue
                except KeyError:
                    logging.info(f'{uid}: no prompt')
                    continue

                logging.info(f'RECV {uid}: "{prompt}"')

                generate(prompt, _path_from_uid(uid), model, sampler, sample_path=None)
        else:
            logging.info('Thread sees no new messages -- sleeping')
            time.sleep(RETRY_DELAY)

def main():
    s3 = boto3.client('s3', region_name=environ['AWS_REGION'])
    sqs = boto3.client('sqs', region_name=environ['AWS_REGION'])

    # TODO: accept environment variables for 
    model, sampler = setup(environ['MODEL_PATH'], False)

    threads = [
                Thread(target=process_queue, args=(i+1, s3, sqs, model, sampler))
                for i in range(int(environ['N_THREADS']))
              ]

    for thread in threads:
        thread.start()

if __name__ == '__main__':
    main()