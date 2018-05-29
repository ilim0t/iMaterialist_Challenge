# iMaterialist_Challenge
1. [train.json](https://www.kaggle.com/c/imaterialist-challenge-fashion-2018/data)をダウンロードします
2. READMEがある階層に移動します
3. 1でダウンロードしたtrain.jsonをこの階層下のinputディレクトリに移動させます
4. train.py を実行

### train.pyの追加した主なオプション
* ***-d***: 画像のダウンロードも同時に行います
* ***-m*** [int]: 以下の内からモデルを選択します

0. ResNet: 18層 RES net
1. ResNet_lite: 上の正規化サイズ128px用
2. Bottle_neck_RES_net: ボトルネックなブロックに変更したRES net
3. Bottle_neck_RES_net_lite: 上の正規化サイズ128px用
4. Mymodel: 簡単なCNN
5. RES_SPP_net: SPP net (実装途中でbatchsize1のときのみ有効)
6. Lite: 動作確認用

* ***-s*** [int]: 正規化する時の1辺の長さ
    指定しないと256px
    モデル名にliteが付くとき128を指定してください
* ***-c*** :真っ白い写真などの余分なデータも学習用データに含めます

run.py -r [snapshotのパス]で data/test_imagesのデータを推測して,result.csvに出力します。学習時と同じオプションを付けてください