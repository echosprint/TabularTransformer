# Variables
PYTHON := python
TRAINSCRIPT := train.py
PREDICTSCRIPT := predict.py

.PHONY: pretrain pretrain1 pretrain2 finetune train predict test clean


ROUND1_CONTRAST_ARGS = --init_from=scratch --data_file=income/income_evaluation_train \
					   --loss_type=4 --pretext_with_label=True \
					   --pretext_target_col=" education" \
					   --output_dim=64 --batch_size=128 \
					   --wandb_log=True

ROUND2_CONTRAST_ARGS = --init_from=resume_with_new_head --data_file=income/income_evaluation_train \
					   --loss_type=4 --pretext_with_label=True \
					   --pretext_target_col=" occupation" \
					   --output_dim=64 --batch_size=128 \
					   --wandb_log=True

PRETRAIN_ARGS = --init_from=scratch --data_file=income/income_evaluation_train \
					   --loss_type=4 --pretext_with_label=True \
					   --pretext_target_col=" occupation" \
					   --output_dim=16 --batch_size=256 \
					   --wandb_log=True --unk_ratio=0.2 \
					   --dim=64 --n_layers=6 --n_heads=8

FINETUNE_ARGS = --init_from=resume_with_new_head --data_file=income/income_evaluation_validate \
					   --loss_type=1 \
					   --output_dim=1 --batch_size=32 \
					   --wandb_log=True \
			   		   --eval_interval=2 --max_iters=500 \
					   --finetune=True --validate_split=0.98 \
					   --dim=64 --n_layers=6 --n_heads=8 \
					   --learning_rate=5e-5 --eval_iters=50 \
					   --checkpoint="pretrained_ckpt.pt"

PLAINTRAIN_ARGS = --init_from=scratch --data_file=income/income_evaluation_validate \
					   --loss_type=1 \
					   --output_dim=1 --batch_size=32 \
					   --wandb_log=True --validate_split=0.98\
			   		   --eval_interval=2 --max_iters=2000 \
					   --dim=64 --n_layers=6 --n_heads=8 \
					   --learning_rate=1e-4 --eval_iters=50 \

TRAIN_ARGS = --init_from=scratch --data_file=income/income_evaluation_train \
					   --loss_type=1 \
					   --output_dim=1 --batch_size=128 \
					   --wandb_log=True --validate_split=0.8\
			   		   --eval_interval=100 --max_iters=5000 \
					   --learning_rate=1e-4 --eval_iters=50 \
					   --dim=64 --n_layers=6 --n_heads=8

PREDICT_ARGS = --predict_dataset="income/income_evaluation_validate.csv"

TEST_ARGS = --init_from=scratch --data_file=income/income_evaluation_train \
					   --loss_type=4 --pretext_with_label=True \
					   --pretext_target_col=" education" \
					   --output_dim=64 --batch_size=128 \
					   --wandb_log=False --eval_only=True


pretrain:
	@echo "Running $(TRAINSCRIPT)..."
	$(PYTHON) $(TRAINSCRIPT) $(PRETRAIN_ARGS)

train:
	@echo "Running $(TRAINSCRIPT)..."
	$(PYTHON) $(TRAINSCRIPT) $(TRAIN_ARGS)

pretrain1:
	@echo "Running $(TRAINSCRIPT)..."
	$(PYTHON) $(TRAINSCRIPT) $(ROUND1_CONTRAST_ARGS)

pretrain2:
	@echo "Running $(TRAINSCRIPT)..."
	$(PYTHON) $(TRAINSCRIPT) $(ROUND2_CONTRAST_ARGS)

finetune:
	@echo "Running $(TRAINSCRIPT)..."
	$(PYTHON) $(TRAINSCRIPT) $(FINETUNE_ARGS)

predict:
	@echo "Running $(PREDICTSCRIPT)..."
	$(PYTHON) $(PREDICTSCRIPT) $(PREDICT_ARGS)

test:
	@echo "Running $(TRAINSCRIPT)..."
	$(PYTHON) $(TRAINSCRIPT) $(TEST_ARGS)

plaintrain:
	@echo "Running $(TRAINSCRIPT)..."
	$(PYTHON) $(TRAINSCRIPT) $(PLAINTRAIN_ARGS)

nbconvert:
	jupyter nbconvert --to markdown income_analysis.ipynb --output-dir=./docs

# Clean target (optional)
clean:
	@echo "Cleaning up..."
	rm -rf __pycache__
