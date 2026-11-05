# GTUD
PROGRAMMING PART
# @title
#-----------TASK 1-----------------
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

#Chọn hằng số A từ -99 đến 80 (A khác 0)
#Chọn A=5
A=5
print(f"Hằng số đã chọn A = {A}")

#khai báo x
x=sp.Symbol('x')

#Định nghĩa hàm g(x) = x^2 - 2Ax - A^2
g_expr = x**2 - 2*A*x - A**2
print(f"Hàm số g(x) = {g_expr} ")

#Tính đạo hàm g'(x)
g_prime = sp.diff(g_expr, x)
print(f"Đạo hàm g'(x) = {g_prime}")

#---------------------------------------------------------
#a) Tiếp tuyến tại (0, -A^2) và giao điểm với đồ thị dời
#---------------------------------------------------------
print("----Task 1.a----")

# x0 = 0, y0 = -A^2
x0_a = 0
y0_a = - A**2

#Tính hệ số góc k tại x0 = 0
slope_a = g_prime.subs(x, x0_a)
print(f"Hệ số góc k tại x = 0: {slope_a}")

#Pttt (T): y - y0 = g'(x0)(x - x0) -> y = g'(x0)(x - x0) + y0
T_a = slope_a * (x - x0_a) + y0_a
print(f"Phương trình tiếp tuyến tại (0, {y0_a}) (T): y = {T_a}")

#Dời đồ thị g(x) xuống 4A^2 đơn vị
g_sau_expr = g_expr - 4*A**2
print(f"Hàm số g(x) sau khi đẩy xuống 4A^2 đơn vị: g1(x) = {g_sau_expr}")

#Tìm hoành độ giao điểm giữa g1(x) và (T)
#Cho g1(x) = T
hdgd_a= sp.solve(g_sau_expr - T_a, x)
print(f"Hoành độ giao điểm giữa g1(x) và (T): x = {hdgd_a}")

#Các giao điểm
cacgiaodiem_a = []
for xv in hdgd_a:
    yv = T_a.subs(x, xv)
    cacgiaodiem_a.append((float(xv), float(yv)))

print(f"Các giao điểm: {cacgiaodiem_a}")
#-------------------------------------
#b) Tiếp tuyến đi qua điểm (0, -4A^2)
#-------------------------------------
print("----Task 1.b----")

x0_b = sp.Symbol('x0')

y0_b = -4*A**2

#Nhắc lại các hàm
#g(x) = x^2 - 2Ax - A^2
#g'(x) = 2*x - 2A
#y = g'(x0)(x - x0) + y0

#Hệ số góc k tại x0 = 0
slope_b = g_prime.subs(x, x0_b)
print(f"Hệ số góc k tại x = 0: {slope_b}")

T_b = slope_b* (x - x0_b) + y0_b
print(f"Phương trình tiếp tuyến tại điểm (0, {y0_b}) (T1): y = {T_b}")

#Lập phương trình theo ẩn x0_b
g_x0_b = g_expr.subs(x, x0_b)
gp_x0_b = g_prime.subs(x, x0_b)

eq_T_b = sp.Eq(y0_b, g_x0_b + gp_x0_b * (0 - x0_b))

#Giải phương trình tọa độ tiếp điểm x0_b
x0_solution = sp.solve(eq_T_b, x0_b)
print(f"Các hoành độ tiếp điểm x0 tương ứng: {x0_solution}")

#-------------------------------------------------
#VẼ ĐỒ THỊ CHO TASK 1 (a,b)
#-------------------------------------------------
print("PHẦN ĐỒ THỊ TASK 1")

# Chuyển đổi các biểu thức sympy sang dạng hàm số numpy để vẽ đồ thị bằng matplotlib
g_func = sp.lambdify(x, g_expr, 'numpy')
g_sau_func = sp.lambdify(x, g_sau_expr, 'numpy')
T_func = sp.lambdify(x, T_a, 'numpy')

#Miền giá trị trục Ox để vẽ đồ thị
x_vals = np.linspace(-3*abs(A) - 5, 3*abs(A) + 5, 400)

#---------------Hình 1.a)-----------------
plt.figure(figsize=(9, 6))
plt.plot(x_vals, g_func(x_vals), label='g(x)', color = 'blue')
plt.plot(x_vals, g_sau_func(x_vals), label='g1(x)', color = 'orange', linestyle = '--')
plt.plot(x_vals, T_func(x_vals), label='T(x)', color = 'green')

#Đánh dấu tiếp điểm (0, -A^2)
plt.scatter(0, -A**2, color='red', label='Điểm (0, -A^2)')

#Đánh dấu các điểm giao nhau
for xv, yv in cacgiaodiem_a:
    plt.scatter(xv, yv, color='purple', s = 75, zorder = 5)
    plt.text(xv, yv, f"({xv:.2f}, {yv:.2f})", fontsize = 9, verticalalignment = 'bottom')

plt.title(f"Task 1.a: Tiếp tuyến và giao điểm đồ thị dời (A = {A})")
plt.xlabel("Ox")
plt.ylabel("Oy")
plt.axhline(0, color = 'black', linewidth = 0.5, linestyle = '-')
plt.axvline(0, color = 'black', linewidth = 0.5, linestyle = '-')
plt.grid(True, linestyle = ':', alpha = 0.7)
plt.legend()
plt.savefig("task1a_detailed.png", dpi = 300)

#---------------Hình 1.b)-----------------
plt.figure(figsize=(9, 6))
plt.plot(x_vals, g_func(x_vals), label='g(x)', color = 'blue')
plt.scatter([0], [-4*A**2], color = 'red', s = 80, zorder = 5, label = 'Điểm cho trước (0, -4A^2)')

# Vẽ các đường tiếp tuyến đi qua điểm (0, -4A^2)
for xv in x0_solution:
    slope_val = gp_x0_b.subs(x0_b, xv)
    t_line_expr = slope_val * (x - xv) + y0_b
    t_line_func = sp.lambdify(x, t_line_expr, 'numpy')

    # Tìm tọa độ y thực tế của tiếp điểm trên đường cong g(x)
    yv_val = g_expr.subs(x, xv)

    plt.plot(x_vals, t_line_func(x_vals), linestyle = '-.', label = f"Tiếp tuyến tại x0 = {float(xv):.2f}")
    plt.scatter([float(xv)], [float(yv_val)], color = 'black', zorder = 5)

plt.title(f"Task 1.b: Các tiếp tuyến đi qua điểm (0, -4A^2) (A = {A})")
plt.xlabel("Ox")
plt.ylabel("Oy")
plt.axhline(0, color = 'black', linewidth = 0.5, linestyle = '-')
plt.axvline(0, color = 'black', linewidth = 0.5, linestyle = '-')
plt.grid(True, linestyle = ':', alpha = 0.7)
plt.legend()
plt.savefig("task1task1b_detailed.png", dpi = 300)

# @title
#-----------Bước 1: Khai báo thư viện và định nghĩa hàm số------------

import math
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

#Chọn hằng số A và tính a = A / 10
A = 15
a = A / 10
print(f"Hằng số A đã chọn là: {A}")

#Khai báo biến x bằng sympy để tính đạo hàm
x = sp.Symbol('x', real = True)
f_expr = (x-a)**2 / 10.0 + 3 * sp.cos(x)

#Chuyển đổi biểu thức sang numpy để tính toán và vẽ đồ thị
f_func = sp.lambdify(x, f_expr, 'numpy')

#Sử dụng sympy để tính đạo hàm
f_prime_expr = sp.diff(f_expr, x)
f_prime_func = sp.lambdify(x, f_prime_expr, 'numpy')

#-------------Bước 2: Xây dựng hàm thuật toán nội suy Parabol----------------

def parabolic_min(f, x1, x2, x3, eps, max_i):
    x4_list = [] #Danh sách lưu lại tất cả các điểm x4 qua các vòng lặp
    k = 1

    while k <= max_i:
        #Tính giá trị hàm số tại 3 điểm
        f1, f2, f3 = f(x1), f(x2), f(x3)

        #Áp dụng công thức (*) để tính hoành độ đỉnh Parabol x4
        TuSo = (x2 - x1)**2 * (f2 - f3) - (x2 - x3)**2 * (f2 - f1)
        MauSo = (x2 - x1) * (f2 - f3) - (x2 - x3) * (f2 - f1)

        if MauSo == 0:
            print("Mẫu số bằng 0, dừng thuật toán")
            break

        x4 = x2 - 0.5 * (TuSo / MauSo)
        x4_list.append(x4)
        f4 = f(x4)

        #In kết quả từng bước lặp định dạng chính xác 8 chữ số thập phân
        print(f"Lần lặp {k} : x = {x4:.8f}, f(x) = {f4:.8f}")

        #Kiểm tra điều kiện dừng nếu khoảng cách rất nhỏ
        if abs(x4 - x2) < eps:
            return x4, f4, x4_list

        #Cập nhật các hàm dựa theo 4 trường hợp trong đề bài
        if x4 > x2:
            if f4 < f2:
                x1, x2, x3 = x2, x4, x3
            else:
                x1, x2, x3 = x1, x2, x4
        else:
            if f4 < f2:
                x1, x2, x3 = x1, x4, x2
            else:
                x1, x2, x3 = x4, x2, x3
        k += 1

    return x4, f(x4), x4_list

#------------Bước 3: Khởi tạo thông số và chạy thuật toán-----------------

x1_init = math.pi / 2
x2_init = math.pi
x3_init = 3 * math.pi / 2
e_val = 1e-8
N_max = 100

x_min, f_min, x4_history = parabolic_min(f_func, x1_init, x2_init, x3_init, e_val, N_max)

print(f"\nCực tiểu: ({x_min:.8f}, {f_min:.8f}), vòng lặp: {len(x4_history)}")

# Dùng sympy tính và kiểm chứng giá trị đạo hàm tại điểm cực tiểu tìm được xem có gần bằng 0 không
deriv_at_min = f_prime_func(x_min)
print(f"Kiểm chứng đạo hàm f'({x_min:.8f}) = {deriv_at_min:.2e} (rất gần 0)")

#-----------Bước 4: Vẽ đồ thị minh họa-------------------

# Tạo mảng giá trị trục x trên đoạn [0, 2*pi]
x_plot_vals = np.linspace(0, 2 * math.pi, 400)
y_plot_vals = f_func(x_plot_vals)

plt.figure(figsize=(9, 6))
plt.plot(x_plot_vals, y_plot_vals, label='f(x)', color='blue')

# Tính toán Parabol P(x) đi qua 3 điểm ban đầu ở vòng lặp đầu tiên để vẽ minh họa
f1_i, f2_i, f3_i = f_func(x1_init), f_func(x2_init), f_func(x3_init)
X_matran = np.array([
    [1, x1_init, x1_init**2],
    [1, x2_init, x2_init**2],
    [1, x3_init, x3_init**2]
])
Y_vecto = np.array([f1_i, f2_i, f3_i])
c0, c1, c2 = np.linalg.solve(X_matran, Y_vecto)
p_func = lambda x: c0 + c1*x + c2*x**2

plt.plot(x_plot_vals, p_func(x_plot_vals), label='Parabola P(x) (iter 1)', color='red', linestyle='--')
plt.scatter([x1_init, x2_init, x3_init], [f1_i, f2_i, f3_i], color='black', zorder=5, label='3 điểm ban đầu')

# Đánh dấu điểm cực tiểu tìm được
plt.scatter([x_min], [f_min], color='green', marker='*', s=150, zorder=7, label=f'Cực tiểu ({x_min:.4f}, {f_min:.4f})')

# Trang trí đồ thị
plt.title(f"Task 2: Cực tiểu nội suy Parabol (A = {A})")
plt.xlabel("x")
plt.ylabel("y")
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()
plt.savefig("task2_result.png", dpi=300)

