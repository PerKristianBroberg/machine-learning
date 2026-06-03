#!/bin/bash
uvicorn api_server.app:app --host 0.0.0.0 --port $PORT
