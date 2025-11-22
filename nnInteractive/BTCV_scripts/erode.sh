for rad in 1 3 4 5 6 7 8 9 10
do
    python predict_btcv.py --mask_aug erode --radius $rad
done