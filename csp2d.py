import os
import sys
import time
import tkinter as tk
from abc import abstractmethod
from tkinter import filedialog, messagebox, ttk

import numpy as np

# Global variable to count export times
export_count = 0


class Policy:
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def get_action(self, observation, info):
        pass

    def _get_stock_size_(self, stock):
        stock_w = np.sum(np.any(stock != -2, axis=1))
        stock_h = np.sum(np.any(stock != -2, axis=0))
        return stock_w, stock_h

    def _can_place_(self, stock, position, prod_size):
        pos_x, pos_y = position
        prod_w, prod_h = prod_size
        return np.all(stock[pos_x : pos_x + prod_w, pos_y : pos_y + prod_h] == -1)


class Policy103(Policy):
    def __init__(self):
        super().__init__()
        self.ListOpt = []
        self.currentIndex = 0

    # Improved Priority Heuristic policy
    def evaluatePriority(self, stock_W, stock_H, observation):
        list_prods = observation["products"]
        prioritized_list = []
        for prod in list_prods:
            if prod["quantity"] > 0:
                prod_size = prod["size"]
                prod_w, prod_h = prod_size
                if prod_w == stock_W and prod_h == stock_H:
                    priority_1 = 4
                elif prod_h == stock_H and prod_w < stock_W:
                    priority_1 = 3
                elif prod_w == stock_W and prod_h < stock_H:
                    priority_1 = 2
                elif prod_w < stock_W and prod_h < stock_H:
                    priority_1 = 1
                else:
                    priority_1 = 0
                # rotate
                prod_w, prod_h = prod_size[::-1]

                if prod_w == stock_W and prod_h == stock_H:
                    priority_2 = 4
                elif prod_h == stock_H and prod_w < stock_W:
                    priority_2 = 3
                elif prod_w == stock_W and prod_h < stock_H:
                    priority_2 = 2
                elif prod_w < stock_W and prod_h < stock_H:
                    priority_2 = 1
                else:
                    priority_2 = 0
                if priority_1 >= priority_2:
                    priority = priority_1
                else:
                    priority = priority_2
                    prod["size"] = prod_size[::-1]
                prioritized_list.append((prod, priority))
        return prioritized_list

    def recursivePacking(self, x, y, stock_W, stock_H, observation, stock_idx=0, m=1):
        results = []
        if stock_W <= 0 or stock_H <= 0:
            return results
        if all(prod["quantity"] == 0 for prod in observation["products"]):
            return results
        priority_list = self.evaluatePriority(stock_W, stock_H, observation)
        priority_list.sort(key=lambda p: (p[1], p[0]["size"][1]), reverse=True)
        for prod, priority in priority_list:
            if prod["quantity"] == 0:
                continue
            prod_w, prod_h = prod["size"]
            if priority == 4:
                results.append(
                    {
                        "stock_idx": stock_idx,
                        "size": (prod_w, prod_h),
                        "position": (x, y),
                    }
                )
                prod["quantity"] -= 1
                break
            elif priority == 3:
                results.append(
                    {
                        "stock_idx": stock_idx,
                        "size": (prod_w, prod_h),
                        "position": (x, y),
                    }
                )
                prod["quantity"] -= 1
                results.extend(
                    self.recursivePacking(
                        x + prod_w,
                        y,
                        stock_W - prod_w,
                        stock_H,
                        observation,
                        stock_idx,
                        m,
                    )
                )
                break
            elif priority == 2:
                results.append(
                    {
                        "stock_idx": stock_idx,
                        "size": (prod_w, prod_h),
                        "position": (x, y),
                    }
                )
                prod["quantity"] -= 1
                results.extend(
                    self.recursivePacking(
                        x,
                        y + prod_h,
                        stock_W,
                        stock_H - prod_h,
                        observation,
                        stock_idx,
                        m,
                    )
                )
                break
            elif priority == 1:
                results.append(
                    {
                        "stock_idx": stock_idx,
                        "size": (prod_w, prod_h),
                        "position": (x, y),
                    }
                )
                prod["quantity"] -= 1
                min_w = min(
                    (
                        p["size"][0]
                        for p in observation["products"]
                        if p["quantity"] > 0
                    ),
                    default=0,
                )
                min_h = min(
                    (
                        p["size"][1]
                        for p in observation["products"]
                        if p["quantity"] > 0
                    ),
                    default=0,
                )
                if stock_W - prod_w < min_w:
                    results.extend(
                        self.recursivePacking(
                            x + prod_w,
                            y,
                            stock_W - prod_w,
                            prod_h,
                            observation,
                            stock_idx,
                            m,
                        )
                    )
                    results.extend(
                        self.recursivePacking(
                            x,
                            y + prod_h,
                            stock_W,
                            stock_H - prod_h,
                            observation,
                            stock_idx,
                            m,
                        )
                    )
                elif stock_H - prod_h < min_h:
                    results.extend(
                        self.recursivePacking(
                            x,
                            y + prod_h,
                            prod_w,
                            stock_H - prod_h,
                            observation,
                            stock_idx,
                            m,
                        )
                    )
                    results.extend(
                        self.recursivePacking(
                            x + prod_w,
                            y,
                            stock_W - prod_w,
                            stock_H,
                            observation,
                            stock_idx,
                            m,
                        )
                    )
                else:
                    hb2 = (stock_W - prod_w) * prod_h
                    vb1 = prod_w * (stock_H - prod_h)
                    if hb2 / vb1 >= m:
                        results.extend(
                            self.recursivePacking(
                                x,
                                y + prod_h,
                                prod_w,
                                stock_H - prod_h,
                                observation,
                                stock_idx,
                                m,
                            )
                        )
                        results.extend(
                            self.recursivePacking(
                                x + prod_w,
                                y,
                                stock_W - prod_w,
                                stock_H,
                                observation,
                                stock_idx,
                                m,
                            )
                        )
                    else:
                        results.extend(
                            self.recursivePacking(
                                x + prod_w,
                                y,
                                stock_W - prod_w,
                                prod_h,
                                observation,
                                stock_idx,
                                m,
                            )
                        )
                        results.extend(
                            self.recursivePacking(
                                x,
                                y + prod_h,
                                stock_W,
                                stock_H - prod_h,
                                observation,
                                stock_idx,
                                m,
                            )
                        )
                break
        return results

    def store_actions(self, observation):
        """Calculates and stores the optimal actions in `ListOpt`."""
        listQuantity = [prod["quantity"] for prod in observation["products"]]
        stockUsed = 1000
        mOp = 1
        stocks = [
            (
                idx,
                np.array(stock, dtype=np.int32),
                self._get_stock_size_(np.array(stock, dtype=np.int32)),
            )
            for idx, stock in enumerate(observation["stocks"])
        ]
        sorted_stocks = sorted(
            stocks,
            key=lambda x: x[2][0] * x[2][1],
            reverse=True,
        )
        for m in range(1, 50):
            indexList = []
            for i, stock, (stock_w, stock_h) in sorted_stocks:
                stock_W, stock_H = self._get_stock_size_(stock)
                indexList.extend(
                    self.recursivePacking(0, 0, stock_W, stock_H, observation, i, m)
                )
                if all(prod["quantity"] == 0 for prod in observation["products"]):
                    if stockUsed > i:
                        stockUsed = i
                        mOp = m
                        break
            for i, prod in enumerate(observation["products"]):
                prod["quantity"] = listQuantity[i]
            indexList.clear()
        self.ListOpt = []
        for i, stock, (stock_w, stock_h) in sorted_stocks:
            stock_W, stock_H = self._get_stock_size_(stock)
            self.ListOpt.extend(
                self.recursivePacking(0, 0, stock_W, stock_H, observation, i, mOp)
            )
        for i, prod in enumerate(observation["products"]):
            prod["quantity"] = listQuantity[i]

    def get_action(self, observation, info):
        """Returns the next action from the precomputed `ListOpt`."""
        if not self.ListOpt:
            self.store_actions(
                observation
            )  # Compute actions if not already computed
        if self.currentIndex < len(self.ListOpt):
            action = self.ListOpt[self.currentIndex]
            self.currentIndex += 1
            return action
        # Return None when all actions are exhausted
        else:
            self.currentIndex = 0
            self.ListOpt.clear()
            self.store_actions(observation)
            action = self.ListOpt[self.currentIndex]
            self.currentIndex += 1
            return action


# CSP Guide
def run_simulation(stocks, products):
    policy103 = Policy103()
    used_stocks = set()
    observation = {"stocks": stocks, "products": products}
    info = {}

    stock_report = [
        {"placed_products": {}, "remaining_area": 0, "used_area": 0, "ratio": 0}
        for _ in stocks
    ]

    while True:
        if len(used_stocks) == len(observation["stocks"]):
            break

        can_place_any = False
        for product in observation["products"]:
            if product["quantity"] == 0:
                continue
            prod_w, prod_h = product["size"]
            for idx, stock in enumerate(observation["stocks"]):
                if idx in used_stocks:
                    continue
                stock_w, stock_h = policy103._get_stock_size_(stock)
                valid_positions = [
                    (x, y)
                    for x in range(stock_w - prod_w + 1)
                    for y in range(stock_h - prod_h + 1)
                    if policy103._can_place_(stock, (x, y), (prod_w, prod_h))
                ]
                if valid_positions:
                    can_place_any = True
                    break
            if can_place_any:
                break

        if not can_place_any:
            break

        action = policy103.get_action(observation, info)
        if action["stock_idx"] == -1:
            break

        stock_idx = action["stock_idx"]
        if stock_idx in used_stocks:
            continue

        size = action["size"]
        pos_x, pos_y = action["position"]
        prod_w, prod_h = size

        observation["stocks"][stock_idx][
            pos_x : pos_x + prod_w, pos_y : pos_y + prod_h
        ] = 0

        stock_data = stock_report[stock_idx]
        product_key = f"{prod_w}x{prod_h}"
        if product_key not in stock_data["placed_products"]:
            stock_data["placed_products"][product_key] = 0
        stock_data["placed_products"][product_key] += 1

        if not np.any(observation["stocks"][stock_idx] == -1):
            used_stocks.add(stock_idx)

        for product in observation["products"]:
            if product["size"] == size:
                product["quantity"] -= 1
                break

    for idx, stock in enumerate(observation["stocks"]):
        remaining_area = np.sum(stock == -1)
        stock_report[idx]["remaining_area"] = remaining_area
        stock_report[idx]["used_area"] = stock.size - remaining_area
        stock_report[idx]["ratio"] = stock_report[idx]["used_area"] / stock.size * 100

    return stock_report, observation["products"]


def display_results(stock_report, remaining_products, stock_count):
    result = "Final stock report:\n"
    stock_used = 0
    for idx, report in enumerate(stock_report):
        if report["used_area"] != 0:
            result += f"Stock {idx+1}:\n"
            for product, quantity in report["placed_products"].items():
                result += f"  - {quantity} product(s) of size {product}\n"
            result += f"  Remaining area: {report['remaining_area']} units\n"
            result += f"  Used area: {report['used_area']} units\n"
            result += f"  Used ratio: {report['ratio']}%\n\n"
            stock_used += 1

    result += "Remaining products:\n"
    for product in remaining_products:
        result += f"Product size {product['size']}: {product['quantity']} remaining\n"
    result += f"Stock used: {stock_used} / {stock_count}\n"
    return result


def load_file():
    file_path = filedialog.askopenfilename(
        title="Select Combined File", filetypes=(("Text Files", "*.txt"),)
    )
    if file_path:
        try:
            with open(file_path, "r") as file:
                content = file.readlines()

                # Tách dữ liệu dựa trên tiêu đề
                stocks, products = [], []
                current_section = None
                is_valid = False

                for line in content:
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith("# Stocks") or line.startswith("#Stocks"):
                        current_section = "stocks"
                        is_valid = True  # Xác nhận có phần # Stocks
                        continue
                    elif line.startswith("# Products") or line.startswith("#Products"):
                        current_section = "products"
                        is_valid = True  # Xác nhận có phần # Products
                        continue

                    if current_section == "stocks":
                        stocks.append(line)
                    elif current_section == "products":
                        products.append(line)

                # Kiểm tra tính hợp lệ
                if not is_valid:
                    raise ValueError(
                        "File is not in correct format! File must have a title # Stocks and # Products."
                    )

                # Nhập dữ liệu vào các ô
                stocks_text.delete("1.0", tk.END)
                stocks_text.insert(tk.END, "\n".join(stocks))

                products_text.delete("1.0", tk.END)
                products_text.insert(tk.END, "\n".join(products))

                messagebox.showinfo("Success", "File loaded successfully!")
        except Exception as e:
            messagebox.showerror(
                "Error",
                f"Can not read file: {e}\n\nRequired file format, for example:\n"
                "# Stocks\n"
                "10x10\n"
                "5x20\n"
                "# Products\n"
                "1x2:10\n"
                "2x3:5",
            )


def export_results_auto():
    global export_count
    export_count += 1
    # Nếu chạy dưới dạng .exe, sử dụng thư mục chứa file .exe
    if hasattr(sys, "_MEIPASS"):
        base_path = os.path.dirname(sys.executable)  # Thư mục chứa file .exe
    else:
        base_path = os.path.dirname(__file__)  # Thư mục script Python

    folder_path = os.path.join(base_path, "exports")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    file_name = os.path.join(folder_path, f"results_{export_count}.txt")
    try:
        with open(file_name, "w") as file:
            file.write(result_text.get("1.0", tk.END))
        messagebox.showinfo("Success", f"Results exported successfully as {file_name}!")
    except Exception as e:
        messagebox.showerror("Error", f"Unable to save file: {e}")


def start_simulation():
    try:
        start_time = time.time()
        stocks_input = stocks_text.get("1.0", tk.END).strip().split("\n")
        products_input = products_text.get("1.0", tk.END).strip().split("\n")

        stocks = []
        for stock in stocks_input:
            dims = list(map(int, stock.split("x")))
            stocks.append(np.full((dims[0], dims[1]), -1))

        products = []
        for product in products_input:
            size, quantity = product.split(":")
            dims = list(map(int, size.split("x")))
            quantity = int(quantity)
            products.append({"size": tuple(dims), "quantity": quantity})

        stock_report, remaining_products = run_simulation(stocks, products)
        result = display_results(stock_report, remaining_products, len(stocks))
        end_time = time.time()  # Ghi lại thời gian kết thúc
        execution_time = end_time - start_time  # Tính thời gian thực thi
        result += f"\nExecution Time: {execution_time:.2f} seconds\n"  # Thêm thời gian vào kết quả
        result_text.delete("1.0", tk.END)
        result_text.insert(tk.END, result)
    except Exception as e:
        messagebox.showerror("Error", f"Invalid input format: {e}")


def center_window(root, width, height):
    # Lấy kích thước màn hình
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    # Tính toán vị trí để căn giữa
    x = (screen_width - width) // 2
    y = (screen_height - height) // 2

    # Đặt vị trí cửa sổ
    root.geometry(f"{width}x{height}+{x}+{y-30}")


def setup_scrollable_text(root, height, width):
    # Frame để chứa Text và Scrollbar
    frame = tk.Frame(root)
    frame.pack(pady=10)

    # Widget Text
    text_widget = tk.Text(frame, height=height, width=width, wrap="word")
    text_widget.pack(side="left", fill="both", expand=True)

    # Scrollbar
    scrollbar = tk.Scrollbar(frame, command=text_widget.yview)
    scrollbar.pack(side="right", fill="y")

    # Liên kết Text và Scrollbar
    text_widget.config(yscrollcommand=scrollbar.set)

    return text_widget


# GUI
root = tk.Tk()
# Gọi hàm để căn giữa cửa sổ, kích thước 800x600
center_window(root, 600, 800)
root.title("CSP2d Guide")
# Instructions
instructions = tk.Label(
    root,
    text="Enter stocks as WIDTH x HEIGHT (e.g., 10x10) one per line.",
)
instructions.pack(pady=10)
# Stock input
stocks_label = tk.Label(root, text="Stocks:")
stocks_label.pack()
stocks_text = setup_scrollable_text(root, height=5, width=40)
stocks_text.pack()

# Instructions
instructions = tk.Label(
    root,
    text="Enter products as WIDTH x HEIGHT : QUANTITY (e.g., 1x2:19) one per line.",
)
instructions.pack(pady=10)
# Product input
products_label = tk.Label(root, text="Products:")
products_label.pack()
products_text = setup_scrollable_text(root, height=5, width=40)
products_text.pack()
# Load file button
load_stock_button = tk.Button(root, text="Load Data File", command=load_file)
load_stock_button.pack(pady=5)
# Run button
run_button = tk.Button(root, text="Run Simulation", command=start_simulation)
run_button.pack(pady=10)

# Result display
result_label = tk.Label(root, text="Results:")
result_label.pack()
result_text = setup_scrollable_text(root, height=15, width=60)
result_text.pack()

export_button = tk.Button(root, text="Export Results", command=export_results_auto)
export_button.pack(pady=5)

root.mainloop()
