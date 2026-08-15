.PHONY: build up down shell test smoke clean

build:
	docker compose build

up:
	docker compose up -d

down:
	docker compose down

shell: up
	docker compose exec geomamba bash

test: up
	docker compose exec -T geomamba pytest

smoke: up
	docker compose exec -T geomamba sh -c "test -f data/smoke_oscillators.npz || python scripts/generate_data.py --smoke"
	docker compose exec -T geomamba python scripts/train.py --config configs/exp/smoke.yaml

clean:
	docker compose down --rmi local
