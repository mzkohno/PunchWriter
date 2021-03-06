# PunchWriter

これを使えば運動しながら仕事ができる！？<br>
運動しながら文章が書けるプログラム！<br>
毎日イスに座ってコーディングばかりしている、仕事人間のあなたにピッタリのコードです。<br>

# Version
2021.10.6<br>
version 0.1.0<br>

# Requirement 
* mediapipe 0.8.6 or later
* OpenCV 3.4.2 or later
* pyokaka 1.0.0 or later

# 動作を確認したOS
Ubuntu 20.04.3 LTS (64bit)

macOSでは動作しない
参考：AppleのUIフレームワークでは、メインスレッド以外からUIを表示することを許可していない
https://developer.apple.com/forums/thread/659010

# 最初の準備

PunchWriter.py コード中のグローバル変数 NIHONGO_FONT を、お使いの環境に合わせて変更してください。<br>
日本語フォントのパスであればOKです。<br>

# 操作方法

### 起動
```
$ python3 PunchWriter.py
```

### 記入モード
・右側パンチ（NEXT）　：次の文字へ<br>
・左側パンチ（BACK）　：前の文字へ<br>
・下パンチ　　　　　　：文字のアセットの変更<br>
・上パンチ（片手）　　：現在の選択文字の仮決定<br>
・上パンチ（両手同時）：仮決定した文字のテキストボックスへの出力<br>
・ヒザ蹴り　　　　　　：仮決定した文字から最後の一文字削除<br>
・バンザイポーズ　　　：テキストボックスの改行<br>
・スクワット　　　　　：変換モードに変更（仮決定した文字がある場合）<br>

### 変換モード
・上パンチ　　　　　　：現在の変換候補のテキストボックスへの出力<br>
・ヒザ蹴り　　　　　　：現在の変換候補の削除<br>
・スクワット　　　　　：次の変換候補へ<br>
すべて出力or削除でき次第、記入モードへ戻る<br>
<br>
＊どちらのモードでも、毎回両手を中央のHitBoxに戻さない限り、次の操作ができない仕様となっている。<br>

### 終了
ESCキー押下でカメラキャプチャを終了します。<br>
テキストウインドウはCLOSEボタンから終了してください。<br>
＊Xボタンで終了しないこと。<br>

# Author
大野 瑞紀（https://qiita.com/Moh_no）
 
# License 
PunchWriter is under Apache-2.0 License.
