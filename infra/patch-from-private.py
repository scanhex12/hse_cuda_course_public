#!/usr/bin/env python3
"""
Подтягивает эталонные решения/тесты из private/ в директории задач.
Запускать из корня репозитория. В каждой задаче должен быть task.json или .tester.json.
Директории с задачами — поддиректории корня (00-hello-world, 01-..., и т.д.)
с наличием task.json или .tester.json.
"""
import os
import shutil
import json

TASK_CONFIG_NAMES = ("task.json", ".tester.json")
PRIVATE_FOLDER_NAME = "./private"


def is_task_dir(path):
    for name in TASK_CONFIG_NAMES:
        if os.path.isfile(os.path.join(path, name)):
            return True
    return False


def patch():
    if not os.path.isdir(PRIVATE_FOLDER_NAME):
        print(f"Directory {PRIVATE_FOLDER_NAME} not found, nothing to patch.")
        return

    for name in sorted(os.listdir(".")):
        if name.startswith(".") or name in ("docs", "tools", "deadlines", "infra", "contrib"):
            continue
        path = os.path.join(".", name)
        if not os.path.isdir(path):
            continue
        if not is_task_dir(path):
            continue
        task_name = name
        private_solution_path = os.path.join(PRIVATE_FOLDER_NAME, task_name)
        if not os.path.isdir(private_solution_path):
            print(f"Task '{task_name}' has no private folder at {private_solution_path}, skip.")
            continue
        print(f"Patching task '{task_name}' from {private_solution_path}")
        shutil.copytree(private_solution_path, path, dirs_exist_ok=True)


if __name__ == "__main__":
    patch()
