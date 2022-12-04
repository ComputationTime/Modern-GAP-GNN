touch ./results.txt
./venv/bin/python3 ./main.py --aggregation_module_name pma  --dataset_name reddit   --epsilon 1 >> ./results.txt
./venv/bin/python3 ./main.py --aggregation_module_name pma  --dataset_name reddit   --epsilon 2 >> ./results.txt
./venv/bin/python3 ./main.py --aggregation_module_name pma  --dataset_name reddit   --epsilon 4 >> ./results.txt
./venv/bin/python3 ./main.py --aggregation_module_name pma  --dataset_name reddit   --epsilon 8 >> ./results.txt
./venv/bin/python3 ./main.py --aggregation_module_name pmwa --dataset_name reddit   --epsilon 1 >> ./results.txt
./venv/bin/python3 ./main.py --aggregation_module_name pmwa --dataset_name reddit   --epsilon 2 >> ./results.txt
./venv/bin/python3 ./main.py --aggregation_module_name pmwa --dataset_name reddit   --epsilon 4 >> ./results.txt
./venv/bin/python3 ./main.py --aggregation_module_name pmwa --dataset_name reddit   --epsilon 8 >> ./results.txt
./venv/bin/python3 ./main.py --aggregation_module_name pma  --dataset_name facebook --epsilon 1 --encoder_hidden_dims "[64]" --encoder_output_dim 32 --classifier_base_output_dim 16 >> ./results.txt
./venv/bin/python3 ./main.py --aggregation_module_name pma  --dataset_name facebook --epsilon 2 --encoder_hidden_dims "[64]" --encoder_output_dim 32 --classifier_base_output_dim 16 >> ./results.txt
./venv/bin/python3 ./main.py --aggregation_module_name pma  --dataset_name facebook --epsilon 4 --encoder_hidden_dims "[64]" --encoder_output_dim 32 --classifier_base_output_dim 16 >> ./results.txt
./venv/bin/python3 ./main.py --aggregation_module_name pma  --dataset_name facebook --epsilon 8 --encoder_hidden_dims "[64]" --encoder_output_dim 32 --classifier_base_output_dim 16 >> ./results.txt
./venv/bin/python3 ./main.py --aggregation_module_name pmwa --dataset_name facebook --epsilon 1 --encoder_hidden_dims "[64]" --encoder_output_dim 32 --classifier_base_output_dim 16 >> ./results.txt
./venv/bin/python3 ./main.py --aggregation_module_name pmwa --dataset_name facebook --epsilon 2 --encoder_hidden_dims "[64]" --encoder_output_dim 32 --classifier_base_output_dim 16 >> ./results.txt
./venv/bin/python3 ./main.py --aggregation_module_name pmwa --dataset_name facebook --epsilon 4 --encoder_hidden_dims "[64]" --encoder_output_dim 32 --classifier_base_output_dim 16 >> ./results.txt
./venv/bin/python3 ./main.py --aggregation_module_name pmwa --dataset_name facebook --epsilon 8 --encoder_hidden_dims "[64]" --encoder_output_dim 32 --classifier_base_output_dim 16 >> ./results.txt