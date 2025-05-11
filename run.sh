python -u train.py -g 0 --dataset="pog" --model="CaRo" --item_augment="FN" --bundle_augment="ID" --bundle_ratio=0.5 --bundle_cl_temp=0.01 --bundle_cl_alpha=0.5 --cl_temp=0.5 --cl_alpha=2 --cate_filter=False

python -u train.py -g 0 --dataset="electronic" --model="CaRo" --item_augment="FN" --bundle_augment="ID" --bundle_ratio=0.5 --bundle_cl_temp=0.01 --bundle_cl_alpha=0.5 --cl_temp=0.5 --cl_alpha=2 --cate_filter=False

python -u train.py -g 0 --dataset="food" --model="CaRo" --item_augment="FN" --bundle_augment="ID" --bundle_ratio=0.5 --bundle_cl_temp=0.01 --bundle_cl_alpha=0.5 --cl_temp=0.5 --cl_alpha=2 --cate_filter=False