import os
import asyncio
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
from dotenv import load_dotenv
from datetime import datetime
import calendar

# === Настройки ===
load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")
DATA_FILE = "data/data.csv"
os.makedirs("data", exist_ok=True)

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

# === Месяцы на русском ===
months_ru = ["январь","февраль","март","апрель","май","июнь",
             "июль","август","сентябрь","октябрь","ноябрь","декабрь"]

# === Главное меню ===
def main_menu():
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="📊 Прогноз на месяц", callback_data="forecast")],
        [InlineKeyboardButton(text="📊 Прогноз на следующий месяц", callback_data="forecast_next")],
        [InlineKeyboardButton(text="📅 План по дням (распределение)", callback_data="plan_by_days")],
        [InlineKeyboardButton(text="📈 Аналитика", callback_data="analytics")],
        [InlineKeyboardButton(text="➕ Добавить данные месяца", callback_data="add_data")],
        [InlineKeyboardButton(text="🧾 Показать данные", callback_data="show_data")],
        [InlineKeyboardButton(text="🧹 Очистить прогноз", callback_data="clear_forecast")]
    ])

# === /start ===
@dp.message(Command("start"))
async def start(message: types.Message):
    await message.answer(
        "👋 Привет! Я TableTrend — бот для прогноза выручки ресторана.\n\n"
        "Я анализирую данные и строю прогноз на следующий месяц 📅",
        reply_markup=main_menu()
    )

# === Очистка графиков ===
@dp.callback_query(lambda c: c.data == "clear_forecast")
async def clear_forecast(callback: types.CallbackQuery):
    folder = "data"
    deleted = 0
    for file in os.listdir(folder):
        if file.startswith("forecast_") and file.endswith(".png"):
            os.remove(os.path.join(folder, file))
            deleted += 1
    await callback.message.answer(f"🧹 Удалено файлов прогнозов: {deleted}")

# === Показ данных ===
@dp.callback_query(lambda c: c.data == "show_data")
async def show_data(callback: types.CallbackQuery):
    if not os.path.exists(DATA_FILE):
        await callback.message.answer("⚠️ Файл с данными не найден.")
        return
    df = pd.read_csv(DATA_FILE)
    text = f"📅 Последние строки:\n\n{df.tail(10).to_string(index=False)}"
    await callback.message.answer(f"<pre>{text}</pre>", parse_mode="HTML")

# === Добавление новых данных месяца ===
user_inputs = {}

@dp.callback_query(lambda c: c.data == "add_data")
async def add_data(callback: types.CallbackQuery):
    user_id = callback.from_user.id
    user_inputs[user_id] = {}
    await callback.message.answer("🗓 Введите месяц и год в формате: `Октябрь 2025`", parse_mode="Markdown")

@dp.message()
async def handle_message(message: types.Message):
    user_id = message.from_user.id
    if user_id not in user_inputs:
        return
    step = user_inputs[user_id]

    if "month" not in step:
        month_text = message.text.strip()
        month_str = month_text.lower().split()[0]
        if month_str not in months_ru:
            await message.answer("❌ Месяц не распознан. Попробуйте, например: Октябрь 2025")
            return
        step["month"] = month_text
        await message.answer("💰 Введите сумму выручки за месяц (например: 1234567):")
    elif "revenue" not in step:
        try:
            step["revenue"] = float(message.text.strip())
            await message.answer("👥 Введите количество гостей за месяц:")
        except ValueError:
            await message.answer("❌ Введите число, например: 1200000")
    elif "guests" not in step:
        try:
            step["guests"] = int(message.text.strip())
            await message.answer("💳 Введите средний чек (например: 845):")
        except ValueError:
            await message.answer("❌ Введите целое число гостей, например: 1400")
    elif "avg_check" not in step:
        try:
            step["avg_check"] = float(message.text.strip())

            month_text = step["month"]
            month_str = month_text.lower().split()[0]
            month_num = months_ru.index(month_str) + 1
            year = int(month_text.split()[-1])
            ds = datetime(year, month_num, 1).strftime("%Y-%m-%d")

            new_row = pd.DataFrame([{
                "ds": ds,
                "revenue": step["revenue"],
                "guests": step["guests"],
                "avg_check": step["avg_check"]
            }])

            if os.path.exists(DATA_FILE):
                df = pd.read_csv(DATA_FILE)
                df["ds"] = pd.to_datetime(df["ds"])
                if ds in df["ds"].dt.strftime("%Y-%m-%d").values:
                    df.loc[df["ds"].dt.strftime("%Y-%m-%d") == ds, ["revenue","guests","avg_check"]] = step["revenue"], step["guests"], step["avg_check"]
                else:
                    df = pd.concat([df, new_row], ignore_index=True)
            else:
                df = new_row
            df.to_csv(DATA_FILE, index=False)

            await message.answer(
                f"✅ Данные добавлены:\n\n"
                f"🗓 {step['month']}\n"
                f"💰 Выручка: {int(step['revenue']):,} ₽\n"
                f"👥 Гости: {step['guests']}\n"
                f"💳 Средний чек: {int(step['avg_check'])} ₽\n\n"
                f"Теперь можно построить прогноз 👉 /start",
                parse_mode="HTML"
            )
            del user_inputs[user_id]
        except ValueError:
            await message.answer("❌ Введите корректное число для среднего чека.")

# === Функция прогноза с русскими подписями ===
titles = {"revenue":"Выручка","guests":"Гости","avg_check":"Средний чек"}

def forecast_metric(monthly, metric):
    monthly = monthly.sort_values("ds").reset_index(drop=True)

    if metric in ["guests", "avg_check"]:
        next_val = monthly[metric].tail(2).mean()
        last_val = monthly[metric].iloc[-1]
        diff = (next_val - last_val) / last_val * 100
        trend = "📈 Растет" if next_val > last_val else "📉 Падает"

        last_6 = monthly[metric].tail(6)
        y_min = last_6.min()
        y_max = last_6.max()

        plt.figure(figsize=(7,4))
        plt.plot(monthly["ds"], monthly[metric], marker="o", label="Факт")
        plt.scatter(monthly["ds"].iloc[-1], last_val, color='green', s=100, label="Прошлый месяц")
        next_month = monthly["ds"].iloc[-1] + pd.DateOffset(months=1)
        plt.scatter(next_month, next_val, color='orange', s=100, label="Прогноз")
        plt.fill_between([next_month], y_min, y_max, color='orange', alpha=0.2)
        plt.title(f"{titles[metric]}: прогноз на следующий месяц ({trend})")
        plt.xlabel("Месяц")
        plt.ylabel(titles[metric])
        plt.ylim(0, max(monthly[metric].max(), y_max)*1.2)
        plt.legend()
        plt.tight_layout()
        img_path = f"data/forecast_{metric}.png"
        plt.savefig(img_path)
        plt.close()

        caption = (
            f"{titles[metric]}\n\n"
            f"🔮 Прогноз: {int(next_val):,}\n"
            f"📉 Минимум: {int(y_min):,}\n"
            f"📈 Максимум: {int(y_max):,}\n"
            f"📊 Изменение: {diff:+.1f}%\n"
            f"💡 Тренд: {trend}"
        ).replace(",", " ")
        return img_path, caption

    else:
        monthly[f"{metric}_lag1"] = monthly[metric].shift(1)
        monthly[f"{metric}_lag1"].fillna(monthly[metric].mean(), inplace=True)
        model = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)
        model.add_regressor(f"{metric}_lag1")
        df_model = monthly.rename(columns={metric: "y"})[["ds", "y", f"{metric}_lag1"]]
        model.fit(df_model)
        future = model.make_future_dataframe(periods=1, freq="M")
        future[f"{metric}_lag1"] = list(monthly[f"{metric}_lag1"]) + [monthly[metric].iloc[-1]]
        forecast_df = model.predict(future)
        last_val = monthly[metric].iloc[-1]
        next_val = forecast_df["yhat"].iloc[-1]
        diff = (next_val - last_val) / last_val * 100
        trend = "📈 Растет" if next_val > last_val else "📉 Падает"

        plt.figure(figsize=(7,4))
        plt.plot(monthly["ds"], monthly[metric], marker="o", label="Факт")
        plt.plot(forecast_df["ds"], forecast_df["yhat"], "--", label="Прогноз", color="orange")
        plt.fill_between(forecast_df["ds"], forecast_df["yhat_lower"], forecast_df["yhat_upper"], color='orange', alpha=0.2)
        plt.scatter(monthly["ds"].iloc[-1], last_val, color='green', s=100, label="Прошлый месяц")
        plt.title(f"{titles[metric]}: прогноз на следующий месяц ({trend})")
        plt.xlabel("Месяц")
        plt.ylabel(titles[metric])
        plt.ylim(0, max(monthly[metric].max(), forecast_df["yhat"].max())*1.2)
        plt.legend()
        plt.tight_layout()
        img_path = f"data/forecast_{metric}.png"
        plt.savefig(img_path)
        plt.close()

        caption = (
            f"{titles[metric]}\n\n"
            f"🔮 Прогноз: {int(next_val):,}\n"
            f"📉 Минимум: {int(forecast_df['yhat_lower'].iloc[-1]):,}\n"
            f"📈 Максимум: {int(forecast_df['yhat_upper'].iloc[-1]):,}\n"
            f"📊 Изменение: {diff:+.1f}%\n"
            f"💡 Тренд: {trend}"
        ).replace(",", " ")
        return img_path, caption

# === Прогноз на месяц ===
@dp.callback_query(lambda c: c.data == "forecast")
async def forecast(callback: types.CallbackQuery):
    if not os.path.exists(DATA_FILE):
        await callback.message.answer("⚠️ Нет данных для прогноза.")
        return

    df = pd.read_csv(DATA_FILE, parse_dates=["ds"])
    df["ds"] = pd.to_datetime(df["ds"])
    monthly = df.resample("M", on="ds").agg({"revenue":"sum","guests":"sum","avg_check":"mean"}).reset_index()
    monthly = monthly.sort_values("ds").reset_index(drop=True)
    monthly = monthly[["ds","revenue","guests","avg_check"]]

    if len(monthly) < 6:
        await callback.message.answer("⚠️ Недостаточно данных для прогноза (нужно минимум 6 месяцев).")
        return

    metrics = ["revenue", "guests", "avg_check"]
    for metric in metrics:
        img, caption = forecast_metric(monthly, metric)
        await callback.message.answer_photo(photo=types.FSInputFile(img), caption=caption)

# === Прогноз на следующий месяц с последними данными ===
@dp.callback_query(lambda c: c.data == "forecast_next")
async def forecast_next(callback: types.CallbackQuery):
    if not os.path.exists(DATA_FILE):
        await callback.message.answer("⚠️ Нет данных для прогноза.")
        return

    df = pd.read_csv(DATA_FILE, parse_dates=["ds"])
    df["ds"] = pd.to_datetime(df["ds"])
    monthly = df.resample("M", on="ds").agg({"revenue":"sum","guests":"sum","avg_check":"mean"}).reset_index()
    monthly = monthly.sort_values("ds").reset_index(drop=True)

    if len(monthly) < 2:
        await callback.message.answer("⚠️ Недостаточно данных для прогноза (нужно минимум 2 месяца).")
        return

    next_month = monthly["ds"].iloc[-1] + pd.DateOffset(months=1)
    next_month_str = next_month.strftime("%B %Y")

    metrics = ["revenue", "guests", "avg_check"]
    for metric in metrics:
        if metric in ["guests", "avg_check"]:
            next_val = monthly[metric].tail(2).mean()
            last_val = monthly[metric].iloc[-1]
            diff = (next_val - last_val) / last_val * 100
            trend = "📈 Растет" if next_val > last_val else "📉 Падает"

            last_6 = monthly[metric].tail(6)
            y_min = last_6.min()
            y_max = last_6.max()

            plt.figure(figsize=(7,4))
            plt.plot(monthly["ds"], monthly[metric], marker="o", label="Факт")
            plt.scatter(monthly["ds"].iloc[-1], last_val, color='green', s=100, label="Прошлый месяц")
            plt.scatter(next_month, next_val, color='orange', s=100, label="Прогноз")
            plt.fill_between([next_month], y_min, y_max, color='orange', alpha=0.2)
            plt.title(f"{titles[metric]}: прогноз на {next_month_str} ({trend})")
            plt.xlabel("Месяц")
            plt.ylabel(titles[metric])
            plt.ylim(0, max(monthly[metric].max(), y_max)*1.2)
            plt.legend()
            plt.tight_layout()
            img_path = f"data/forecast_next_{metric}.png"
            plt.savefig(img_path)
            plt.close()

            caption = (
                f"{titles[metric]} — прогноз на {next_month_str}\n\n"
                f"🔮 Прогноз: {int(next_val):,}\n"
                f"📉 Минимум: {int(y_min):,}\n"
                f"📈 Максимум: {int(y_max):,}\n"
                f"📊 Изменение: {diff:+.1f}%\n"
                f"💡 Тренд: {trend}"
            ).replace(",", " ")
            await callback.message.answer_photo(photo=types.FSInputFile(img_path), caption=caption)
        else:
            monthly[f"{metric}_lag1"] = monthly[metric].shift(1)
            monthly[f"{metric}_lag1"].fillna(monthly[metric].mean(), inplace=True)
            model = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)
            model.add_regressor(f"{metric}_lag1")
            df_model = monthly.rename(columns={metric: "y"})[["ds", "y", f"{metric}_lag1"]]
            model.fit(df_model)
            future = model.make_future_dataframe(periods=1, freq="M")
            future[f"{metric}_lag1"] = list(monthly[f"{metric}_lag1"]) + [monthly[metric].iloc[-1]]
            forecast_df = model.predict(future)
            last_val = monthly[metric].iloc[-1]
            next_val = forecast_df["yhat"].iloc[-1]
            diff = (next_val - last_val) / last_val * 100
            trend = "📈 Растет" if next_val > last_val else "📉 Падает"

            plt.figure(figsize=(7,4))
            plt.plot(monthly["ds"], monthly[metric], marker="o", label="Факт")
            plt.plot(forecast_df["ds"], forecast_df["yhat"], "--", label="Прогноз", color="orange")
            plt.fill_between(forecast_df["ds"], forecast_df["yhat_lower"], forecast_df["yhat_upper"], color='orange', alpha=0.2)
            plt.scatter(monthly["ds"].iloc[-1], last_val, color='green', s=100, label="Прошлый месяц")
            plt.title(f"{titles[metric]}: прогноз на {next_month_str} ({trend})")
            plt.xlabel("Месяц")
            plt.ylabel(titles[metric])
            plt.ylim(0, max(monthly[metric].max(), forecast_df["yhat"].max())*1.2)
            plt.legend()
            plt.tight_layout()
            img_path = f"data/forecast_next_{metric}.png"
            plt.savefig(img_path)
            plt.close()

            caption = (
                f"{titles[metric]} — прогноз на {next_month_str}\n\n"
                f"🔮 Прогноз: {int(next_val):,}\n"
                f"📉 Минимум: {int(forecast_df['yhat_lower'].iloc[-1]):,}\n"
                f"📈 Максимум: {int(forecast_df['yhat_upper'].iloc[-1]):,}\n"
                f"📊 Изменение: {diff:+.1f}%\n"
                f"💡 Тренд: {trend}"
            ).replace(",", " ")
            await callback.message.answer_photo(photo=types.FSInputFile(img_path), caption=caption)

# === НОВАЯ КНОПКА: План по дням (распределение) ===
@dp.callback_query(lambda c: c.data == "plan_by_days")
async def plan_by_days(callback: types.CallbackQuery):
    """
    Берём тот же monthly, что и в forecast_next.
    Строим прогноз суммы на следующий месяц для revenue (используем ту же логику с lag1).
    Затем распределяем эту сумму по дням месяца по заданным весам и
    отправляем готовый план (картинка + текст), при этом сумма по дням == прогнозной сумме.
    """
    if not os.path.exists(DATA_FILE):
        await callback.message.answer("⚠️ Нет данных для прогноза.")
        return

    df = pd.read_csv(DATA_FILE, parse_dates=["ds"])
    df["ds"] = pd.to_datetime(df["ds"])
    monthly = df.resample("M", on="ds").agg({"revenue":"sum","guests":"sum","avg_check":"mean"}).reset_index()
    monthly = monthly.sort_values("ds").reset_index(drop=True)

    if len(monthly) < 2:
        await callback.message.answer("⚠️ Недостаточно данных для построения плана (нужно минимум 2 месяца).")
        return

    # --- Получаем прогнозную сумму для следующего месяца (как в forecast_next для revenue) ---
    metric = "revenue"
    monthly[f"{metric}_lag1"] = monthly[metric].shift(1)
    monthly[f"{metric}_lag1"].fillna(monthly[metric].mean(), inplace=True)

    model = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)
    model.add_regressor(f"{metric}_lag1")
    df_model = monthly.rename(columns={metric: "y"})[["ds", "y", f"{metric}_lag1"]]
    model.fit(df_model)
    future = model.make_future_dataframe(periods=1, freq="M")
    future[f"{metric}_lag1"] = list(monthly[f"{metric}_lag1"]) + [monthly[metric].iloc[-1]]
    forecast_df = model.predict(future)

    next_month_val = float(forecast_df["yhat"].iloc[-1])  # итоговая прогнозная сумма для месяца
    next_month = monthly["ds"].iloc[-1] + pd.DateOffset(months=1)
    next_month_str = next_month.strftime("%B %Y")

    # --- Генерируем все даты следующего месяца ---
    year = next_month.year
    month = next_month.month
    days_in_month = calendar.monthrange(year, month)[1]
    dates = pd.date_range(start=f"{year}-{month:02d}-01", periods=days_in_month, freq="D")

    # --- Вес для дней недели (можно настроить) ---
    # 0=Пн .. 6=Вс
    weekday_weights = {
        0: 0.95,  # Пн
        1: 1.00,  # Вт
        2: 1.00,  # Ср
        3: 1.05,  # Чт
        4: 1.10,  # Пт
        5: 1.30,  # Сб
        6: 1.20   # Вс
    }

    plan_df = pd.DataFrame({"ds": dates})
    plan_df["weekday"] = plan_df["ds"].dt.weekday
    plan_df["weight"] = plan_df["weekday"].map(weekday_weights).fillna(1.0)

    # Нормализуем веса и умножаем на forecast сумму
    total_weight = plan_df["weight"].sum()
    plan_df["revenue_plan_raw"] = plan_df["weight"] / total_weight * next_month_val

    # Округляем до целых рублей и корректируем, чтобы сумма точно совпала с next_month_val
    plan_df["revenue_plan"] = plan_df["revenue_plan_raw"].round().astype(int)
    diff = int(round(next_month_val)) - int(plan_df["revenue_plan"].sum())
    if diff != 0:
        # Корректируем: добавляем ±1 к дням с наибольшим весом (если положительный diff) или убираем (если отрицательный)
        # Сортируем по весу и применяем правку
        idx_sorted = plan_df.sort_values("weight", ascending=False).index
        i = 0
        step = 1 if diff > 0 else -1
        diff_abs = abs(diff)
        while diff_abs > 0:
            plan_df.at[idx_sorted[i % len(idx_sorted)], "revenue_plan"] += step
            diff_abs -= 1
            i += 1

    # Финальная проверка — приводим сумму в строку для вывода
    total_plan = int(plan_df["revenue_plan"].sum())

    # --- Построим график плана по дням ---
    plt.figure(figsize=(10, 4.5))
    plt.plot(plan_df["ds"], plan_df["revenue_plan"], marker="o", linewidth=1)
    plt.title(f"План выручки по дням — {next_month_str}")
    plt.xlabel("Дата")
    plt.ylabel("Выручка (₽)")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    img_path = f"data/forecast_plan_by_days_{year}_{month}.png"
    plt.savefig(img_path)
    plt.close()

    # --- Текстовое представление (короткая таблица) ---
    text_lines = [f"📅 План выручки на {next_month_str} (итого: {total_plan:,} ₽)\n"]
    for _, r in plan_df.iterrows():
        weekday_name = r["ds"].strftime("%a")  # короткое имя дня
        text_lines.append(f"{r['ds'].strftime('%d.%m.%Y')} ({weekday_name}) — {int(r['revenue_plan']):,} ₽")
    text = "\n".join(text_lines).replace(",", " ")

    # Отправляем картинку + текст
    await callback.message.answer_photo(photo=types.FSInputFile(img_path), caption=text)

# === Аналитика ===
@dp.callback_query(lambda c: c.data == "analytics")
async def analytics(callback: types.CallbackQuery):
    if not os.path.exists(DATA_FILE):
        await callback.message.answer("⚠️ Нет данных для анализа.")
        return

    df = pd.read_csv(DATA_FILE, parse_dates=["ds"])
    monthly = df.resample("M", on="ds").agg({"revenue":"sum","guests":"sum","avg_check":"mean"}).reset_index()

    avg_rev = monthly["revenue"].mean()
    avg_guests = monthly["guests"].mean()
    avg_check = monthly["avg_check"].mean()
    best_month = monthly.loc[monthly["revenue"].idxmax(), "ds"].strftime("%B %Y")
    worst_month = monthly.loc[monthly["revenue"].idxmin(), "ds"].strftime("%B %Y")

    text = (
        f"📊 Аналитика по месяцам:\n\n"
        f"💰 Средняя выручка: {int(avg_rev):,} ₽\n"
        f"👥 Среднее кол-во гостей: {int(avg_guests)}\n"
        f"💳 Средний чек: {int(avg_check)} ₽\n\n"
        f"🏆 Лучший месяц: {best_month}\n"
        f"📉 Слабейший месяц: {worst_month}"
    ).replace(",", " ")

    await callback.message.answer(text)

# === Запуск ===
async def main():
    print("✅ TableTrend запущен!")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())

