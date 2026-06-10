# Деплой Astra ML на сервер НГТУ (cloud.nstu.ru)

Цель: поднять Streamlit-приложение на VDS НГТУ так, чтобы на защите можно было
показать **QR-код → рабочий проект в браузере**.

> Гайд преподавателя описывает универсальный VDS (Apache для статики и Python-сокеты).
> У нас **Streamlit**, поэтому раздел «Apache раздаёт HTML» и «сокетный сервер» не нужны —
> вместо них Streamlit крутится как системный сервис (демон). Этап «создать ВМ + зайти по SSH»
> берётся из гайда без изменений.

---

## 0. Что заказать при создании ВМ

В гайде НГТУ для учебной ВМ предлагают 256–512 МБ ОЗУ / 1 vCPU. **Этого мало** —
scikit-learn + pandas + matplotlib + plotly при обучении модели съедят больше.

- **ОЗУ:** минимум 1 ГБ, лучше 2 ГБ.
- **vCPU:** 1–2.
- **Диск:** 10–15 ГБ.
- **ОС:** Ubuntu Server 22.04/20.04 LTS (без GUI — легче и быстрее).

---

## 1. Создать ВМ и зайти по SSH  *(по гайду НГТУ, без изменений)*

1. Зарегистрироваться на `cloud.nstu.ru`, создать проект и сеть.
2. Создать ВМ (Ubuntu Server LTS), задать root-пароль.
3. Поднять OpenVPN-подключение по гайду и подключиться к ВМ:
   ```
   ssh -p <порт> root@ssh.cloud.nstu.ru
   ```
4. Обновить пакеты:
   ```bash
   apt-get update && apt-get upgrade -y
   ```
5. (Рекомендуется гайдом) создать sudo-пользователя `astra` вместо работы под root:
   ```bash
   adduser astra
   usermod -aG sudo astra
   su astra
   ```
   Если решите остаться под root — в `astra.service` поставьте `User=root` и путь `/root/astra`.

---

## 2. Поставить Python и зависимости

```bash
sudo apt-get install -y python3 python3-venv python3-pip git
```

---

## 3. Залить проект на сервер

**Вариант A — через git** (если репозиторий доступен с сервера):
```bash
sudo mkdir -p /opt/astra && sudo chown astra:astra /opt/astra
git clone <URL-репозитория> /opt/astra
```

**Вариант B — через scp с вашего компьютера** (как в гайде НГТУ).
В PowerShell на ноутбуке, из корня проекта `D:\Developing\Astra`:
```powershell
scp -P <порт> -r D:\Developing\Astra\* astra@ssh.cloud.nstu.ru:/opt/astra/
```
> Не копируйте `.venv\` — виртуальное окружение для Windows на Linux не заработает,
> его создадим заново на сервере (шаг 4). Достаточно `web_app.py`, `src/`, `data/`,
> `requirements.txt`, `.streamlit/`.

---

## 4. Создать окружение и установить пакеты

```bash
cd /opt/astra
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Проверить, что приложение вообще стартует:
```bash
streamlit run web_app.py --server.address 0.0.0.0 --server.port 8501
```
Должно написать `You can now view your Streamlit app ... Network URL: http://<IP>:8501`.
Останавливаем `Ctrl+C` — дальше запустим как сервис.

---

## 5. Запустить как системный сервис (демон)

Чтобы приложение не падало при выходе из SSH и поднималось после перезагрузки:

```bash
sudo cp /opt/astra/deploy/astra.service /etc/systemd/system/astra.service
sudo systemctl daemon-reload
sudo systemctl enable --now astra
sudo systemctl status astra        # должно быть active (running)
```
Логи в реальном времени:
```bash
journalctl -u astra -f
```

---

## 6. Открыть порт в фаерволе (UFW)

```bash
sudo ufw allow 22       # чтобы не потерять SSH
sudo ufw allow 8501     # порт Streamlit
sudo ufw enable
sudo ufw status
```

> **Важно (специфика облака НГТУ):** кроме UFW внутри ВМ, порт нужно открыть
> и в **группе безопасности / правилах сети** в панели cloud.nstu.ru
> (по аналогии с тем, как в гайде открывают TCP-порт для сокетного сервера:
> «Протокол TCP; порт; направление входящее»). Без этого правила снаружи порт
> будет закрыт, даже если UFW его разрешает.

---

## 7. Узнать публичный IP и собрать QR

Узнать IP, который виден снаружи:
```bash
ip a            # внутренний адрес ВМ
curl ifconfig.me   # внешний IP (если есть выход в интернет)
```

Адрес для QR: `http://<ВНЕШНИЙ_IP>:8501`

Сгенерировать QR можно прямо на ноутбуке (PowerShell, в venv проекта):
```powershell
pip install qrcode[pil]
python -c "import qrcode; qrcode.make('http://<ВНЕШНИЙ_IP>:8501').save('astra_qr.png')"
```
Либо любым онлайн-генератором QR по этой ссылке.

---

## 8. (Опционально) Чистый URL без :8501 через Apache

Если хочется, чтобы QR вёл на `http://<IP>` без порта — поднять обратный прокси Apache:

```bash
sudo apt-get install -y apache2
sudo a2enmod proxy proxy_http proxy_wstunnel rewrite
sudo cp /opt/astra/deploy/astra-apache.conf /etc/apache2/sites-available/astra.conf
sudo a2dissite 000-default.conf
sudo a2ensite astra.conf
sudo apache2ctl configtest
sudo systemctl restart apache2
sudo ufw allow 80
```
Теперь QR делаем на `http://<ВНЕШНИЙ_IP>` (порт 80). Конфиг уже учитывает WebSocket
Streamlit (`/_stcore/stream`) — без этого интерфейс «вечно грузится».

---

## ⚠️ Проверить ДО защиты

1. **Доступность снаружи.** Откройте `http://<IP>:8501` **с телефона по мобильному
   интернету** (не по Wi-Fi НГТУ). Если облако НГТУ даёт адрес, видимый только внутри
   университетской сети, — снаружи QR не откроется. Это надо выяснить заранее: либо адрес
   реально публичный, либо защиту показывать с устройства в сети НГТУ, либо использовать
   запасной вариант (ниже).
2. **Сервис автозапускается** после `sudo reboot` (`systemctl status astra`).
3. **Загрузка CSV** работает (лимит 50 МБ в `.streamlit/config.toml`).
4. **Память.** Прогоните обучение на демо-датасете и посмотрите `htop` — хватает ли ОЗУ.

---

## 🅱️ Запасной вариант (если НГТУ-сервер недоступен снаружи / не успели)

### Streamlit Community Cloud — самый простой, бесплатный, идеален для QR
1. Залить проект на **GitHub** (публичный репозиторий; `requirements.txt` у вас уже есть).
2. Зайти на <https://share.streamlit.io>, войти через GitHub.
3. «New app» → выбрать репозиторий, ветку и файл `web_app.py` → Deploy.
4. Через пару минут получите постоянный **HTTPS-URL** вида
   `https://<имя>.streamlit.app` — на него и делаете QR.

Плюсы: публичный HTTPS-адрес из коробки, ничего настраивать не надо, работает с любого
устройства. Минус: загруженные CSV сохраняются на их сервере (для демо не важно).

### Быстрый туннель с ноутбука (если приложение крутится локально)
На крайний случай — пробросить локальный Streamlit наружу одной командой:
```powershell
# вариант 1: cloudflared (без регистрации)
cloudflared tunnel --url http://localhost:8501
# вариант 2: ngrok
ngrok http 8501
```
Команда выдаст публичный HTTPS-URL → QR. Работает, только пока открыт ноутбук и команда.
Хорошо как «страховка в кармане» на самой защите.
