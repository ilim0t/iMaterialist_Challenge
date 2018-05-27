# iMaterialist_Challenge
1. [script.py](https://www.kaggle.com/nlecoy/imaterialist-downloader-util)をダウンロードします
2. [train.json](https://www.kaggle.com/c/imaterialist-challenge-fashion-2018/data)をダウンロードします
3. READMEがある階層に移動します
4. ダウンロードしたtrain.jsonをこの階層下のinputフォルダに追加します
5. python [script.pyのパス] [train.jsonのパス] data/train_images と実行
6. train.py を実行


script.pyの63行目
parse_dataset関数への第三引数を与えればよりたくさんダウンロードできます。

run.py -r [snapshotのパス]で data/test_imagesのデータを推測して,result.csvに出力します。