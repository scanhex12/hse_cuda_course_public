# Настройка окружения

Для решения задач нужен Unix-подобное окружение (Linux или macOS). Для Windows можно использовать WSL2.

## Регистрация и репозиторий

1. Зарегистрируйтесь в системе курса (notmanytask / GitLab по инструкции преподавателя).
2. Для вас создаётся личный репозиторий. Сохраните его адрес (Clone with SSH).

## Локальный репозиторий

1. Склонируйте **публичный репозиторий курса** с задачами:
   ```bash
   git clone <URL репозитория курса>
   cd hse_cuda_course_public   # или как называется директория
   ```

2. Настройте git (если ещё не настроен):
   ```bash
   git config --global user.name "Имя Фамилия"
   git config --global user.email "your@email.com"
   ```

3. Добавьте свой **личный репозиторий** как remote `student`:
   ```bash
   git remote add student <URL вашего личного репо>
   ```
   URL скопируйте со страницы репозитория (Clone → Clone with SSH).

## Сборка и запуск тестов

Из корня репозитория курса:

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --target test_00_hello_world
./test_00_hello_world
```

Или из директории задачи (если в ней есть CMake):

```bash
cd 00-hello-world
# сборка через корневой CMake или как указано в задаче
```

## Отправка решения

Запускайте скрипт **из директории задачи**:

```bash
cd 00-hello-world
python3 ../submit.py
```

Либо из корня, указав задачу:

```bash
python3 submit.py 00-hello-world
```

Далее смотрите статус в CI/CD → Pipelines вашего личного репозитория и в веб-интерфейсе курса.
