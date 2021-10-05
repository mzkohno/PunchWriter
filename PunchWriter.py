#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import time
import os
import argparse
import cv2 as cv
import mediapipe as mp
import tkinter as tk
from tkinter.scrolledtext import ScrolledText
import threading

import requests
from pyokaka import okaka

import numpy as np
from PIL import Image, ImageDraw, ImageFont

################################ グローバル変数 ################################
NIHONGO_FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc" # 日本語フォントのパス

TYPE_CHAR = "" # tkinterにopenCVから文字列を渡すための変数（別スレッド）

################################ ローマ字かな漢字変換 ################################
def romaTranslater(word):
    # ローマ字をひらがなに変換（pyokaka）
    yomi_hira = okaka.convert(word)
    #print(yomi_hira)

    # ひらがなを漢字に変換（google翻訳にrequests.get）
    url = "http://www.google.com/transliterate?"
    param = {'langpair':'ja-Hira|ja','text':yomi_hira}

    res = requests.get(url=url, params=param)
    if res.status_code != 200:
        print("EROOR: 漢字変換できません。ネットワークに接続してください")
        return yomi_hira
    res = res.json()
    return res

################################ OpenCVで日本語を書くための関数 ################################
# OpenCVは日本語非対応のため、Pillow(PIL)で描画してimageを返すクラス
# https://qiita.com/Kazuhito/items/6a030b3746d941c3d9c7 を参考に設定
class CvPutJaText:
    def __init__(self):
        pass

    @classmethod
    def puttext(cls, cv_image, text, point, font_path, font_size, color=(0,0,0), stroke_color=(255,255,255)):
        font = ImageFont.truetype(font_path, font_size)

        cv_rgb_image = cv.cvtColor(cv_image, cv.COLOR_BGR2RGB)
        pil_image = Image.fromarray(cv_rgb_image)

        draw = ImageDraw.Draw(pil_image)
        draw.text(point, text, fill=color, font=font, stroke_width=1, stroke_fill=stroke_color)

        cv_rgb_result_image = np.asarray(pil_image)
        cv_bgr_result_image = cv.cvtColor(cv_rgb_result_image, cv.COLOR_RGB2BGR)

        return cv_bgr_result_image

################################ tkinterクラス ################################
class Application(tk.Frame):
    def __init__(self,master):
        super().__init__(master)
        self.pack()

        self.width=480 # 横幅
        self.height=680 # 縦幅
        self.event = threading.Event()
        self.checkNumber=1

        self.isTranslating = False
        self.getJson = ""

        master.geometry(str(self.width)+"x"+str(self.height)) #ウィンドウの作成

        master.title("PunchWriter") #タイトル
        self.master.config(bg="white") #ウィンドウの背景色

        self.createWidgets() #ウィジェットの作成

    #*************************** テキスト入力制御 ***************************#
    
    # change_valueを別スレッドで実行するコールバック
    def change_value_callback(self):
        self.th = threading.Thread(target=self.change_value, args=())
        self.th.start()

    # TYPE_CHARに文字が入っている場合、self.textにinsertして文字をクリアする
    def change_value(self):
        global TYPE_CHAR
        while not self.event.wait(0.5):
            time.sleep(0.05)
            ### 仮テキストボックスへの記入 ###
            if TYPE_CHAR != "":
                #print(TYPE_CHAR)
                self.text.insert(tk.END, TYPE_CHAR)
                TYPE_CHAR = ""
    #**************************************************************************#

    # 終了ボタン
    def fin_button(self):
        self.event.set()
        time.sleep(0.5)
        self.th.join()
        # print(threading.enumerate()) # 起動中のスレッド確認用デバックコード
        self.master.destroy()

    """# BackSpaceボタン
    def back_button(self):
        lenText = len(self.text.get("1.0", tk.END))
        print(self.text.get("0.0", tk.END))
        print(lenText)
        if lenText > 0:
            self.text.delete("{0}.0".format(lenText-2), tk.END)"""

    # ラベルウィジェット作成
    def createWidgets(self):
        label = tk.Label(
            self,
            text="PunchWriter",
            width=15,
            height=1
        )
        label.pack()

        self.text = ScrolledText(self, font=("", 30), height=9, width=50)
        self.text.pack()
        self.change_value_callback()

        # ボタンの作成と配置
        """self.button1 = tk.Button(
            text="BACKSPACE",  # 初期値
            width=15,  # 幅
            bg="lightblue",  # 色
            command=self.back_button  # クリックに実行する関数
        )
        self.button1.pack()"""

        self.button2 = tk.Button(
            text="CLOSE",  # 初期値
            width=15,  # 幅
            bg="lightblue",  # 色
            command=self.fin_button  # クリックに実行する関数
        )
        self.button2.pack()

################################ OpenCV ################################
### カメラのキャプチャ
def cameraCap():
    global TYPE_CHAR
    # 引数解析
    args = get_args()

    cap_width = 960 # 横幅
    cap_height = 540 # 縦幅
    cap_device = args.device
    debug = args.debug

    static_image_mode = args.static_image_mode
    model_complexity = args.model_complexity
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    # カメラ準備
    cap = cv.VideoCapture(cap_device)
    camera_opened = cap.isOpened()
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # モデルロード
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=static_image_mode,
        model_complexity=model_complexity,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    ### 以下繰り返しキャプチャする処理
    oc = OpenCV() # インスタンス作成
    while camera_opened:
        # カメラキャプチャ
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  # ミラー表示
        cameraImg = copy.deepcopy(image)

        # 検出実施
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        results = pose.process(image)

        # 描画（画像の作成）
        if results.pose_landmarks is not None:
            cameraImg, decidedChar = oc.draw_landmarks(
                cameraImg,
                results.pose_landmarks,
                debug
            )
        if decidedChar: TYPE_CHAR = decidedChar

        # 作成した画像の画面反映
        cv.imshow('PunchWriter', cameraImg)

        # キー処理
        key = cv.waitKey(1)
        if key == 27:
            # ESC：終了
            cap.release()
            cv.destroyAllWindows()
            return

    cap.release()
    cv.destroyAllWindows()

# 動画中繰り返し実行される画像処理のクラス
class OpenCV:
    def __init__(self):
        self.cmdLock = {"nose":True, "hand": True, "enter":True, "knee":True}
        self.tmpSent = ""

        # 記入用変数
        self.cmdSetIndex = 0
        self.cmdCharIndex = 0
        self.cmdSetList = ["abcdefgh","ijklmnop","qrstuvwxyz","0123456789","=+-*/","()!?,.#"]
        self.tmpChar = self.cmdSetList[0][0]

        # 変換用変数
        self.henkanMode = False # False:記入モード, True:変換モード
        self.henkanSentList = []
        self.henkanCandiNo = 0

    ### 変換系変数の初期化
    def initHenkanVar(self):
        self.tmpChar = self.cmdSetList[0][0]
        self.tmpSent = ""
        self.henkanMode = False
        self.henkanCandiNo = 0

    ### 次の変換センテンスへ移動する
    def nextHenkanSent(self):
        self.henkanCandiNo = 0
        del self.henkanSentList[0]
        if len(self.henkanSentList) > 0: self.tmpSent = self.henkanSentList[0][1][0]

    ### 動きでコマンド指定、図形の描画（Debug:ランドマークの可視化）
    def draw_landmarks(
        self,
        image,
        landmarks,
        debug,
        visibility_th=0.5,
    ):
        image_width, image_height = image.shape[1], image.shape[0]

        landmark_point = []
        decideChar = None

        # landmark_pointの作成（Pointer）
        rowPoints = [0,image_width//3*1,image_width//3*2,image_width//3*3]
        hiPoints = [0,image_height//3*1,image_height//3*2,image_height//3*3]
        balancer = {"row": rowPoints[1]//3, "hi": hiPoints[1]//3}

        for index, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)
            landmark_z = landmark.z
            landmark_point.append([landmark.visibility, (landmark_x, landmark_y)])

            if landmark.visibility < visibility_th:
                continue

            ########################################################
            # debugモードをONにした場合のみ実行（Pointerの可視化）
            if debug:
                # landmarkに小さい円を表示する
                if index == 0:  # 鼻
                    cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
                if index == 1:  # 右目：目頭
                    cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
                if index == 2:  # 右目：瞳
                    cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
                if index == 3:  # 右目：目尻
                    cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
                if index == 4:  # 左目：目頭
                    cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
                if index == 5:  # 左目：瞳
                    cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
                if index == 6:  # 左目：目尻
                    cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
                if index == 7:  # 右耳
                    cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
                if index == 8:  # 左耳
                    cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
                if index == 9:  # 口：左端
                    cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
                if index == 10:  # 口：左端
                    cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
                if index == 11:  # 右肩
                    cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
                if index == 12:  # 左肩
                    cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
                if index == 13:  # 右肘
                    cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
                if index == 14:  # 左肘
                    cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
                if index == 15:  # 右手首
                    cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
                if index == 16:  # 左手首
                    cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
                if index == 17:  # 右手1(外側端)
                    cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
                if index == 18:  # 左手1(外側端)
                    cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
                if index == 19:  # 右手2(先端)
                    cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
                if index == 20:  # 左手2(先端)
                    cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
                if index == 21:  # 右手3(内側端)
                    cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
                if index == 22:  # 左手3(内側端)
                    cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
                if index == 23:  # 腰(右側)
                    cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
                if index == 24:  # 腰(左側)
                    cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
                if index == 25:  # 右ひざ
                    cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
                if index == 26:  # 左ひざ
                    cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
                if index == 27:  # 右足首
                    cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
                if index == 28:  # 左足首
                    cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
                if index == 29:  # 右かかと
                    cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
                if index == 30:  # 左かかと
                    cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
                if index == 31:  # 右つま先
                    cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
                if index == 32:  # 左つま先
                    cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)

                # if not upper_body_only:
                cv.putText(image, "z:" + str(round(landmark_z, 3)),
                    (landmark_x - 10, landmark_y - 10),
                    cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                    cv.LINE_AA)
            ########################################################

        # HitBoxと文字の描画
        ### 変換モード ###
        if self.henkanMode:
            # HitBoxの表示
            cv.rectangle(image, (rowPoints[1]+10, hiPoints[0]+10), (rowPoints[2]-10, hiPoints[1]-balancer["hi"]-10), (255, 0, 0)) # 中央上
            cv.rectangle(image, (rowPoints[1]+10, hiPoints[1]+10), (rowPoints[2]-10, hiPoints[2]-10), (0, 255, 0)) # 中央緑
            cv.rectangle(image, (rowPoints[1]+10, hiPoints[2]+balancer["hi"]+10), (rowPoints[2]-10, hiPoints[3]-10), (255, 0, 0)) # 中央下
            # 文字表示
            # 左上
            image = CvPutJaText.puttext(image, "変換モード", (rowPoints[0]+20, hiPoints[0]+20), NIHONGO_FONT, 25, (0, 0, 0), (255,255,255))
            # 文字列を画像化してimageに重ねる
            # 中央上
            image = CvPutJaText.puttext(image, self.tmpSent, (rowPoints[1]+20, hiPoints[0]+40), NIHONGO_FONT, 30, (0, 0, 0), (255,255,255))
            # 中央下
            image = CvPutJaText.puttext(image, str(self.henkanSentList[0][1]), (rowPoints[1]+20, hiPoints[2]+balancer["hi"]+40), NIHONGO_FONT, 20, (0, 0, 0), (255,255,255))
        ### 記入モード ###
        else:
            # HitBoxの表示
            cv.rectangle(image, (rowPoints[1]+10, hiPoints[0]+10), (rowPoints[2]-10, hiPoints[1]-balancer["hi"]-10), (255, 0, 0)) # 中央上
            cv.rectangle(image, (rowPoints[0]+10, hiPoints[1]+10), (rowPoints[1]-balancer["row"]-10, hiPoints[2]-10), (255, 0, 0)) # 左
            cv.rectangle(image, (rowPoints[1]+10, hiPoints[1]+10), (rowPoints[2]-10, hiPoints[2]-10), (0, 255, 0)) # 中央緑
            cv.rectangle(image, (rowPoints[2]+balancer["row"]+10, hiPoints[1]+10), (rowPoints[3]-10, hiPoints[2]-10), (255, 0, 0)) # 右
            cv.rectangle(image, (rowPoints[1]+10, hiPoints[2]+balancer["hi"]+10), (rowPoints[2]-10, hiPoints[3]-10), (255, 0, 0)) # 中央下
            # 文字表示
            # 左上
            image = CvPutJaText.puttext(image, "記入モード", (rowPoints[0]+20, hiPoints[0]+20), NIHONGO_FONT, 25, (0, 0, 0), (255,255,255))
            # 中央上
            cv.putText(image, self.tmpSent, (rowPoints[1]+20, hiPoints[0]+40),
                cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 3, cv.LINE_AA) # 白枠
            cv.putText(image, self.tmpSent, (rowPoints[1]+20, hiPoints[0]+40),
                cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2, cv.LINE_AA) # 黒字
            # 左中央
            cv.putText(image, "BACK", (rowPoints[0]+20, hiPoints[1]+40),
                cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 3, cv.LINE_AA) # 白枠
            cv.putText(image, "BACK", (rowPoints[0]+20, hiPoints[1]+40),
                cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2, cv.LINE_AA) # 黒字
            # 中央（中央に大きく仮文字を表示）
            cv.putText(image, self.tmpChar, (image_width//2-25, image_height//2+10),
                cv.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 6, cv.LINE_AA)
            cv.putText(image, self.tmpChar, (image_width//2-25, image_height//2+10),
                cv.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 0), 4, cv.LINE_AA)
            # 右中央
            cv.putText(image, "NEXT", (rowPoints[2]+balancer["row"]+20, hiPoints[1]+40),
                cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 3, cv.LINE_AA) # 白枠
            cv.putText(image, "NEXT", (rowPoints[2]+balancer["row"]+20, hiPoints[1]+40),
                cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2, cv.LINE_AA) # 黒字
            # 中央下
            cv.putText(image, self.cmdSetList[self.cmdSetIndex], (rowPoints[1]+20, hiPoints[2]+balancer["hi"]+40),
                cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 3, cv.LINE_AA)
            cv.putText(image, self.cmdSetList[self.cmdSetIndex], (rowPoints[1]+20, hiPoints[2]+balancer["hi"]+40),
                cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2, cv.LINE_AA)

        #*************************** テキスト入力制御 ***************************#
        # 変換 鼻（スクワット）
        if self.henkanMode:
            if self.cmdLock["nose"]:
                if hiPoints[2]+10 >= landmark_point[0][1][1]:
                    self.cmdLock["nose"] = False
            else:
                # 変換
                if hiPoints[2]+10 < landmark_point[0][1][1]:
                    self.henkanCandiNo += 1
                    if self.henkanCandiNo >= len(self.henkanSentList[0][1]):
                        self.henkanCandiNo = 0
                    self.tmpSent = self.henkanSentList[0][1][self.henkanCandiNo]
                    self.cmdLock["nose"] = True
                    cv.rectangle(image, (rowPoints[1]+10, hiPoints[0]+10), (rowPoints[2]-10, hiPoints[1]-balancer["hi"]-10), (0, 0, 255), thickness=5)
        else:
            if hiPoints[2]+10 < landmark_point[0][1][1] and self.tmpSent != "":
                # 変換開始（各種変数の初期化）
                self.henkanSentList = romaTranslater(self.tmpSent)
                #print(self.henkanSentList)
                self.tmpSent = self.henkanSentList[0][1][self.henkanCandiNo]
                self.henkanMode = True
                self.cmdLock["nose"] = True
                cv.rectangle(image, (rowPoints[1]+10, hiPoints[0]+10), (rowPoints[2]-10, hiPoints[1]-balancer["hi"]-10), (0, 0, 255), thickness=5)

        # 文字入力の制御 右手2 or 左手2(先端)
        ### 変換モード ###
        if self.henkanMode:
            if self.cmdLock["hand"]:
                if  (rowPoints[1]+10 < landmark_point[19][1][0] < rowPoints[2]-10 and hiPoints[1]+10 < landmark_point[19][1][1] < hiPoints[2]-10) \
                and (rowPoints[1]+10 < landmark_point[20][1][0] < rowPoints[2]-10 and hiPoints[1]+10 < landmark_point[20][1][1] < hiPoints[2]-10) :
                    self.cmdLock["hand"] = False # ロック解除
                    cv.rectangle(image, (rowPoints[1]+10, hiPoints[1]+10), (rowPoints[2]-10, hiPoints[2]-10), (0, 255, 0), thickness=5)
            else:
                # 上パンチ
                if (rowPoints[1]+10 < landmark_point[19][1][0] < rowPoints[2]-10 and landmark_point[19][1][1] < hiPoints[1]-balancer["hi"]-10) \
                or (rowPoints[1]+10 < landmark_point[20][1][0] < rowPoints[2]-10 and landmark_point[20][1][1] < hiPoints[1]-balancer["hi"]-10) :
                    decideChar = self.tmpSent # 決定
                    self.nextHenkanSent() # 次のセンテンスへ移動する
                    if self.henkanSentList == []:
                        self.initHenkanVar() # 初期化
                    self.cmdLock["hand"] = True
                    cv.rectangle(image, (rowPoints[1]+10, hiPoints[0]+10), (rowPoints[2]-10, hiPoints[1]-balancer["hi"]-10), (0, 0, 255), thickness=5)
        ### 記入モード ###
        else: 
            if self.cmdLock["hand"]:
                if  (rowPoints[1]+10 < landmark_point[19][1][0] < rowPoints[2]-10 and hiPoints[1]+10 < landmark_point[19][1][1] < hiPoints[2]-10) \
                and (rowPoints[1]+10 < landmark_point[20][1][0] < rowPoints[2]-10 and hiPoints[1]+10 < landmark_point[20][1][1] < hiPoints[2]-10) :
                    self.cmdLock["hand"] = False # ロック解除
                    cv.rectangle(image, (rowPoints[1]+10, hiPoints[1]+10), (rowPoints[2]-10, hiPoints[2]-10), (0, 255, 0), thickness=5)
            else:
                # 横パンチ（文字送り）
                if (landmark_point[19][1][0] < rowPoints[1]-balancer["row"]-10 and hiPoints[1]+10 < landmark_point[19][1][1] < hiPoints[2]-10) \
                or (landmark_point[20][1][0] < rowPoints[1]-balancer["row"]-10 and hiPoints[1]+10 < landmark_point[20][1][1] < hiPoints[2]-10) :
                    self.cmdCharIndex -= 1 # 一つ戻る
                    if self.cmdCharIndex < 0:
                        self.cmdCharIndex = len(self.cmdSetList[self.cmdSetIndex])-1
                    self.tmpChar = self.cmdSetList[self.cmdSetIndex][self.cmdCharIndex]
                    self.cmdLock["hand"] = True
                    cv.rectangle(image, (rowPoints[0]+10, hiPoints[1]+10), (rowPoints[1]-balancer["row"]-10, hiPoints[2]-10), (0, 0, 255), thickness=5) # HitBoxフラッシュ
                elif (landmark_point[19][1][0] > rowPoints[2]+balancer["row"]+10 and hiPoints[1]+10 < landmark_point[19][1][1] < hiPoints[2]-10) \
                or   (landmark_point[20][1][0] > rowPoints[2]+balancer["row"]+10 and hiPoints[1]+10 < landmark_point[20][1][1] < hiPoints[2]-10) :
                    self.cmdCharIndex += 1 # 一つ進む
                    if self.cmdCharIndex >= len(self.cmdSetList[self.cmdSetIndex]):
                        self.cmdCharIndex = 0
                    self.tmpChar = self.cmdSetList[self.cmdSetIndex][self.cmdCharIndex]
                    self.cmdLock["hand"] = True
                    cv.rectangle(image, (rowPoints[2]+balancer["row"]+10, hiPoints[1]+10), (rowPoints[3]-10, hiPoints[2]-10), (0, 0, 255), thickness=5) # HitBoxフラッシュ
                # バンザイ（改行）
                elif (rowPoints[2]+10 < landmark_point[19][1][0] and landmark_point[19][1][1] < hiPoints[1]-balancer["hi"]-10) \
                and  (rowPoints[0]+10 < landmark_point[20][1][0] and landmark_point[20][1][1] < hiPoints[1]-balancer["hi"]-10) :
                    decideChar = "\n" # 改行
                    self.cmdLock["hand"] = True
                    cv.rectangle(image, (rowPoints[0]+10, hiPoints[0]+10), (rowPoints[1]-10, hiPoints[1]-balancer["hi"]-10), (0, 0, 255), thickness=5)
                    cv.rectangle(image, (rowPoints[2]+10, hiPoints[0]+10), (rowPoints[3]-10, hiPoints[1]-balancer["hi"]-10), (0, 0, 255), thickness=5)
                # 上パンチ（両手同時でアルファベットのまま本決定、片手のみで仮決定）
                elif (rowPoints[1]+10 < landmark_point[19][1][0] < rowPoints[2]-10 and landmark_point[19][1][1] < hiPoints[1]-balancer["hi"]-10) \
                and  (rowPoints[1]+10 < landmark_point[20][1][0] < rowPoints[2]-10 and landmark_point[20][1][1] < hiPoints[1]-balancer["hi"]-10) :
                    decideChar = self.tmpSent # 決定
                    self.initHenkanVar() # 初期化
                    self.cmdLock["hand"] = True
                    cv.rectangle(image, (rowPoints[1]+10, hiPoints[0]+10), (rowPoints[2]-10, hiPoints[1]-balancer["hi"]-10), (0, 0, 255), thickness=5)
                elif (rowPoints[1]+10 < landmark_point[19][1][0] < rowPoints[2]-10 and landmark_point[19][1][1] < hiPoints[1]-balancer["hi"]-10) \
                or   (rowPoints[1]+10 < landmark_point[20][1][0] < rowPoints[2]-10 and landmark_point[20][1][1] < hiPoints[1]-balancer["hi"]-10) :
                    self.tmpSent += self.tmpChar # 仮決定
                    self.cmdLock["hand"] = True
                    cv.rectangle(image, (rowPoints[1]+10, hiPoints[0]+10), (rowPoints[2]-10, hiPoints[1]-balancer["hi"]-10), (0, 0, 255), thickness=5)
                # 下パンチ
                elif (rowPoints[1]+10 < landmark_point[19][1][0] < rowPoints[2]-10 and landmark_point[19][1][1] > hiPoints[2]+balancer["hi"]+10):
                    self.cmdCharIndex = 0
                    self.cmdSetIndex += 1
                    if self.cmdSetIndex >= len(self.cmdSetList):
                        self.cmdSetIndex = 0
                    self.tmpChar = self.cmdSetList[self.cmdSetIndex][self.cmdCharIndex]
                    self.cmdLock["hand"] = True
                    cv.rectangle(image, (rowPoints[1]+10, hiPoints[2]+balancer["hi"]+10), (rowPoints[2]-10, hiPoints[3]-10), (0, 0, 255), thickness=5)
                elif (rowPoints[1]+10 < landmark_point[20][1][0] < rowPoints[2]-10 and landmark_point[20][1][1] > hiPoints[2]+balancer["hi"]+10):
                    self.cmdCharIndex = 0
                    self.cmdSetIndex -= 1
                    if self.cmdSetIndex < 0:
                        self.cmdSetIndex = len(self.cmdSetList)-1
                    self.tmpChar = self.cmdSetList[self.cmdSetIndex][self.cmdCharIndex]
                    self.cmdLock["hand"] = True
                    cv.rectangle(image, (rowPoints[1]+10, hiPoints[2]+balancer["hi"]+10), (rowPoints[2]-10, hiPoints[3]-10), (0, 0, 255), thickness=5)

        # 文字の削除 ひざ
        ### 変換モード ###
        if self.henkanMode:
            if self.cmdLock["knee"]:
                if  not (rowPoints[1]+10 < landmark_point[25][1][0] < rowPoints[2]-10 and hiPoints[1]+10 < landmark_point[25][1][1] < hiPoints[2]-10) \
                and not (rowPoints[1]+10 < landmark_point[26][1][0] < rowPoints[2]-10 and hiPoints[1]+10 < landmark_point[26][1][1] < hiPoints[2]-10):
                    self.cmdLock["knee"] = False
            else:
                if (rowPoints[1]+10 < landmark_point[25][1][0] < rowPoints[2]-10 and hiPoints[1]+10 < landmark_point[25][1][1] < hiPoints[2]-10) \
                or (rowPoints[1]+10 < landmark_point[26][1][0] < rowPoints[2]-10 and hiPoints[1]+10 < landmark_point[26][1][1] < hiPoints[2]-10):
                    self.nextHenkanSent() # 次のセンテンスへ移動する
                    if self.henkanSentList == []:
                        self.initHenkanVar() # 初期化
                    self.cmdLock["knee"] = True
                    cv.rectangle(image, (rowPoints[1]+10, hiPoints[1]+10), (rowPoints[2]-10, hiPoints[2]-10), (0, 0, 255), thickness=5)
        ### 記入モード ###
        else:
            if self.cmdLock["knee"]:
                if  not (rowPoints[1]+10 < landmark_point[25][1][0] < rowPoints[2]-10 and hiPoints[1]+10 < landmark_point[25][1][1] < hiPoints[2]-10) \
                and not (rowPoints[1]+10 < landmark_point[26][1][0] < rowPoints[2]-10 and hiPoints[1]+10 < landmark_point[26][1][1] < hiPoints[2]-10):
                    self.cmdLock["knee"] = False
            else:
                if (rowPoints[1]+10 < landmark_point[25][1][0] < rowPoints[2]-10 and hiPoints[1]+10 < landmark_point[25][1][1] < hiPoints[2]-10) \
                or (rowPoints[1]+10 < landmark_point[26][1][0] < rowPoints[2]-10 and hiPoints[1]+10 < landmark_point[26][1][1] < hiPoints[2]-10):
                    if self.tmpSent != "": self.tmpSent = self.tmpSent[:-1]
                    self.cmdLock["knee"] = True
                    cv.rectangle(image, (rowPoints[1]+10, hiPoints[1]+10), (rowPoints[2]-10, hiPoints[2]-10), (0, 0, 255), thickness=5)
        #******************************************************#

        ########################################################
        # debugモードをONにした場合のみ表示（landmark同士を線でつなぐ）
        if debug:
            # 右目
            if landmark_point[1][0] > visibility_th and landmark_point[2][0] > visibility_th:
                cv.line(image, landmark_point[1][1], landmark_point[2][1],(0, 255, 0), 2)
            if landmark_point[2][0] > visibility_th and landmark_point[3][0] > visibility_th:
                cv.line(image, landmark_point[2][1], landmark_point[3][1],(0, 255, 0), 2)

            # 左目
            if landmark_point[4][0] > visibility_th and landmark_point[5][0] > visibility_th:
                cv.line(image, landmark_point[4][1], landmark_point[5][1],(0, 255, 0), 2)
            if landmark_point[5][0] > visibility_th and landmark_point[6][0] > visibility_th:
                cv.line(image, landmark_point[5][1], landmark_point[6][1],(0, 255, 0), 2)

            # 口
            if landmark_point[9][0] > visibility_th and landmark_point[10][0] > visibility_th:
                cv.line(image, landmark_point[9][1], landmark_point[10][1],(0, 255, 0), 2)

            # 肩
            if landmark_point[11][0] > visibility_th and landmark_point[12][0] > visibility_th:
                cv.line(image, landmark_point[11][1], landmark_point[12][1],(0, 255, 0), 2)

            # 右腕
            if landmark_point[11][0] > visibility_th and landmark_point[13][0] > visibility_th:
                cv.line(image, landmark_point[11][1], landmark_point[13][1],(0, 255, 0), 2)
            if landmark_point[13][0] > visibility_th and landmark_point[15][0] > visibility_th:
                cv.line(image, landmark_point[13][1], landmark_point[15][1],(0, 255, 0), 2)

            # 左腕
            if landmark_point[12][0] > visibility_th and landmark_point[14][0] > visibility_th:
                cv.line(image, landmark_point[12][1], landmark_point[14][1],(0, 255, 0), 2)
            if landmark_point[14][0] > visibility_th and landmark_point[16][0] > visibility_th:
                cv.line(image, landmark_point[14][1], landmark_point[16][1],(0, 255, 0), 2)

            # 右手
            if landmark_point[15][0] > visibility_th and landmark_point[17][0] > visibility_th:
                cv.line(image, landmark_point[15][1], landmark_point[17][1],(0, 255, 0), 2)
            if landmark_point[17][0] > visibility_th and landmark_point[19][0] > visibility_th:
                cv.line(image, landmark_point[17][1], landmark_point[19][1],(0, 255, 0), 2)
            if landmark_point[19][0] > visibility_th and landmark_point[21][0] > visibility_th:
                cv.line(image, landmark_point[19][1], landmark_point[21][1],(0, 255, 0), 2)
            if landmark_point[21][0] > visibility_th and landmark_point[15][0] > visibility_th:
                cv.line(image, landmark_point[21][1], landmark_point[15][1],(0, 255, 0), 2)

            # 左手
            if landmark_point[16][0] > visibility_th and landmark_point[18][0] > visibility_th:
                cv.line(image, landmark_point[16][1], landmark_point[18][1],(0, 255, 0), 2)
            if landmark_point[18][0] > visibility_th and landmark_point[20][0] > visibility_th:
                cv.line(image, landmark_point[18][1], landmark_point[20][1],(0, 255, 0), 2)
            if landmark_point[20][0] > visibility_th and landmark_point[22][0] > visibility_th:
                cv.line(image, landmark_point[20][1], landmark_point[22][1],(0, 255, 0), 2)
            if landmark_point[22][0] > visibility_th and landmark_point[16][0] > visibility_th:
                cv.line(image, landmark_point[22][1], landmark_point[16][1],(0, 255, 0), 2)

            # 胴体
            if landmark_point[11][0] > visibility_th and landmark_point[23][0] > visibility_th:
                cv.line(image, landmark_point[11][1], landmark_point[23][1],(0, 255, 0), 2)
            if landmark_point[12][0] > visibility_th and landmark_point[24][0] > visibility_th:
                cv.line(image, landmark_point[12][1], landmark_point[24][1],(0, 255, 0), 2)
            if landmark_point[23][0] > visibility_th and landmark_point[24][0] > visibility_th:
                cv.line(image, landmark_point[23][1], landmark_point[24][1],(0, 255, 0), 2)

            if len(landmark_point) > 25:
                # 右足
                if landmark_point[23][0] > visibility_th and landmark_point[25][0] > visibility_th:
                    cv.line(image, landmark_point[23][1], landmark_point[25][1],(0, 255, 0), 2)
                if landmark_point[25][0] > visibility_th and landmark_point[27][0] > visibility_th:
                    cv.line(image, landmark_point[25][1], landmark_point[27][1],(0, 255, 0), 2)
                if landmark_point[27][0] > visibility_th and landmark_point[29][0] > visibility_th:
                    cv.line(image, landmark_point[27][1], landmark_point[29][1],(0, 255, 0), 2)
                if landmark_point[29][0] > visibility_th and landmark_point[31][0] > visibility_th:
                    cv.line(image, landmark_point[29][1], landmark_point[31][1],(0, 255, 0), 2)

                # 左足
                if landmark_point[24][0] > visibility_th and landmark_point[26][0] > visibility_th:
                    cv.line(image, landmark_point[24][1], landmark_point[26][1],(0, 255, 0), 2)
                if landmark_point[26][0] > visibility_th and landmark_point[28][0] > visibility_th:
                    cv.line(image, landmark_point[26][1], landmark_point[28][1],(0, 255, 0), 2)
                if landmark_point[28][0] > visibility_th and landmark_point[30][0] > visibility_th:
                    cv.line(image, landmark_point[28][1], landmark_point[30][1],(0, 255, 0), 2)
                if landmark_point[30][0] > visibility_th and landmark_point[32][0] > visibility_th:
                    cv.line(image, landmark_point[30][1], landmark_point[32][1],(0, 255, 0), 2)
        ########################################################

        return image, decideChar

################################ 引数の受け取り ################################
def get_args():
    parser = argparse.ArgumentParser()

    # debugモードの選択
    parser.add_argument("--debug", type=bool, default=False)

    ###### 指定する機会は少ない ######
    # カメラデバイスの選択
    parser.add_argument("--device", type=int, default=0)
    # mediapipeの設定
    parser.add_argument('--static_image_mode', action='store_true')
    parser.add_argument("--model_complexity",
                        help='model_complexity(0,1(default),2)',
                        type=int,
                        default=1)
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.5)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)
    ###################################

    args = parser.parse_args()

    return args
    
################################ メイン ################################
def main():
    # フォントのパスを自分用に設定していない場合、ここで警告する
    if not os.path.exists(NIHONGO_FONT):
        print("####################################################")
        print("＊日本語用フォントの正しいパスが指定されていません。")
        print("　NIHONGO_FONTにパスを指定してください。")
        print("####################################################")
        return

    # スレッドでOpenCVを起動
    thread1 = threading.Thread(target=cameraCap)
    thread1.start()

    # メインスレッドでtkinterのウインドウを起動
    win = tk.Tk()
    win.resizable(width=False, height=False) #ウィンドウを固定サイズに
    app = Application(master=win)
    app.mainloop()

# 実行
if __name__ == '__main__':
    main()
