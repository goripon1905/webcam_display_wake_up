import cv2
import numpy as np
import pyautogui

# Webカメラの初期化
cap = cv2.VideoCapture(0)

# 前フレームとの差分を計算するための変数を初期化
prev_frame = None

# カメラが接続されているかどうかをチェックするためのフラグ
is_camera_connected = True

# 感度を調整するための変数
SENSITIVITY = 200

# メインループ
while True:
    # カメラから画像を取得
    ret, frame = cap.read()

    # カメラが接続されていない場合はループを抜ける
    if not ret:
        is_camera_connected = False
        break

    # グレースケールに変換
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 最初のフレームの場合は前フレームに現在のフレームをセット
    if prev_frame is None:
        prev_frame = gray

    # 前フレームとの差分を計算
    diff = cv2.absdiff(prev_frame, gray)

    # 二値化処理
    thresh = cv2.threshold(diff, SENSITIVITY, 255, cv2.THRESH_BINARY)[1]

    # 輪郭を取得
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 輪郭を描画
    cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

    # マウスを少しだけ動かして画面スリープを解除
    if len(contours) > 0:
        pyautogui.move(1, 1)

    # 現在のフレームを前フレームにセット
    prev_frame = gray

    # 画像を表示
    cv2.imshow('frame', frame)

    # 'q'キーが押された場合はループを抜ける
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 後処理
if is_camera_connected:
    cap.release()

cv2.destroyAllWindows()
