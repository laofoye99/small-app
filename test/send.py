import json
import requests

API_URL = "https://tennis.madingyu.com/api/admin/SpaceParties/reportData"
TIMEOUT = (3, 10)
HEADERS = {"Content-Type": "application/json"}

PAYLOAD = {
  "serial_number": "FV9942593",
  "startTime": "2026-04-17T02:59:29Z",
  "endTime": "2026-04-17T02:59:35Z",
  "content": {
    "mete": {
      "farCount": {
        "totalShots": 1,
        "avgBallSpeed": 80.0,
        "maxBallSpeed": 80.0,
        "totalDistance": 0.05,
        "avgMoveSpeed": 0.009,
        "maxMoveSpeed": 1.373,
        "firstServeSuccessRate": 0,
        "returnFirstSuccessRate": 0,
        "baselineShotRate": 0,
        "baselineWinRate": 0,
        "netPointRate": 0,
        "netPointWinRate": 0,
        "aceCount": 0,
        "netApproaches": 0
      },
      "nearCount": {
        "totalShots": 1,
        "avgBallSpeed": 80.0,
        "maxBallSpeed": 80.0,
        "totalDistance": 0.06,
        "avgMoveSpeed": 0.009,
        "maxMoveSpeed": 1.405,
        "firstServeSuccessRate": 0,
        "returnFirstSuccessRate": 0,
        "baselineShotRate": 0,
        "baselineWinRate": 0,
        "netPointRate": 0,
        "netPointWinRate": 0,
        "aceCount": 0,
        "netApproaches": 0
      }
    },
    "resultmatrix": [
      {
        "x": 0.0,
        "y": 0.548,
        "type": "hit",
        "speed": 80.0,
        "handType": "forehand"
      }
    ],
    "trackMatrix": [
      {
        "x": 0.0,
        "y": 0.26,
        "type": "running",
        "speed": 0,
        "timestamp": 803,
        "farCountPerson_x": 0.761,
        "farCountPerson_y": 0.507,
        "nearCountPerson_x": 0.057,
        "nearCountPerson_y": 0.09
      },
      {
        "x": 0.0,
        "y": 0.399,
        "type": "running",
        "speed": 0,
        "timestamp": 804,
        "farCountPerson_x": 0.761,
        "farCountPerson_y": 0.51,
        "nearCountPerson_x": 0.061,
        "nearCountPerson_y": 0.09
      },
      {
        "x": 0.0,
        "y": 0.548,
        "type": "running",
        "speed": 0,
        "timestamp": 805,
        "farCountPerson_x": 0.76,
        "farCountPerson_y": 0.512,
        "nearCountPerson_x": 0.066,
        "nearCountPerson_y": 0.089
      },
      {
        "x": 0.0,
        "y": 0.55,
        "type": "running",
        "speed": 0,
        "timestamp": 806,
        "farCountPerson_x": 0.76,
        "farCountPerson_y": 0.512,
        "nearCountPerson_x": 0.066,
        "nearCountPerson_y": 0.089
      },
      {
        "x": 0.0,
        "y": 0.701,
        "type": "running",
        "speed": 0,
        "timestamp": 807,
        "farCountPerson_x": 0.759,
        "farCountPerson_y": 0.513,
        "nearCountPerson_x": 0.066,
        "nearCountPerson_y": 0.089
      },
      {
        "x": 0.0,
        "y": 0.702,
        "type": "running",
        "speed": 0,
        "timestamp": 808,
        "farCountPerson_x": 0.759,
        "farCountPerson_y": 0.514,
        "nearCountPerson_x": 0.067,
        "nearCountPerson_y": 0.09
      },
      {
        "x": 0.0,
        "y": 0.702,
        "type": "running",
        "speed": 0,
        "timestamp": 809,
        "farCountPerson_x": 0.758,
        "farCountPerson_y": 0.515,
        "nearCountPerson_x": 0.067,
        "nearCountPerson_y": 0.09
      },
      {
        "x": 0.0,
        "y": 0.703,
        "type": "running",
        "speed": 0,
        "timestamp": 810,
        "farCountPerson_x": 0.759,
        "farCountPerson_y": 0.515,
        "nearCountPerson_x": 0.067,
        "nearCountPerson_y": 0.09
      },
      {
        "x": 0.0,
        "y": 0.704,
        "type": "running",
        "speed": 0,
        "timestamp": 811,
        "farCountPerson_x": 0.759,
        "farCountPerson_y": 0.515,
        "nearCountPerson_x": 0.067,
        "nearCountPerson_y": 0.09
      },
      {
        "x": 0.0,
        "y": 0.704,
        "type": "running",
        "speed": 0,
        "timestamp": 812,
        "farCountPerson_x": 0.759,
        "farCountPerson_y": 0.515,
        "nearCountPerson_x": 0.067,
        "nearCountPerson_y": 0.09
      }
    ]
  }
}

response = requests.post(url=API_URL,json=PAYLOAD,headers=HEADERS,timeout=TIMEOUT)

response.raise_for_status()

print(f"响应内容：{response.json()}")