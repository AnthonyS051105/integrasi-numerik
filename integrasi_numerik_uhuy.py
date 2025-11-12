import numpy as np

def trapezoidal_rule(f, a, b, n):
    """
    Metode Trapezoidal Rule
    f: fungsi yang akan diintegralkan
    a: batas bawah
    b: batas atas
    n: jumlah subinterval
    """
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)
    
    result = h * (0.5 * y[0] + np.sum(y[1:-1]) + 0.5 * y[-1])
    return result

def richardson_extrapolation(f, a, b, n):
    """
    Richardson extrapolation tingkat pertama untuk meningkatkan
    akurasi hasil integrasi trapezoidal.
    """
    I_h = trapezoidal_rule(f, a, b, n)
    I_h2 = trapezoidal_rule(f, a, b, 2 * n)
    return (4 * I_h2 - I_h) / 3

def romberg_integration(f, a, b, max_level=3):
    """
    Melakukan integrasi numerik menggunakan metode Romberg.
    
    f          : fungsi yang akan diintegralkan
    a, b       : batas bawah dan atas integral
    max_level  : jumlah maksimum level tabel (default 5)
    
    Return:
        R       : tabel Romberg (numpy array)
        integral: hasil aproksimasi terbaik (elemen kanan bawah)
    """
    R = np.zeros((max_level, max_level))
    for i in range(max_level):
        n = 2**i
        R[i, 0] = trapezoidal_rule(f, a, b, n)
    
        for j in range(1, i + 1):
            R[i, j] = R[i, j-1] + (R[i, j-1] - R[i-1, j-1]) / (4**j - 1)

    
    return R, R[max_level-1, max_level-1]

def adaptive_simpson(f, a, b, tol=1e-10, max_depth=50):
    """
    Metode Adaptive Integration menggunakan Simpson's Rule
    """
    def simpson_rule(f, a, b):
        """Simpson's rule untuk interval [a, b]"""
        c = (a + b) / 2
        h = b - a
        return (h / 6) * (f(a) + 4 * f(c) + f(b))
    
    def adaptive_aux(f, a, b, tol, S, fa, fb, fc, depth):
        """Fungsi rekursif untuk adaptive integration"""
        c = (a + b) / 2
        d = (a + c) / 2
        e = (c + b) / 2
        
        fd = f(d)
        fe = f(e)
        
        Sleft = (c - a) / 6 * (fa + 4 * fd + fc)
        Sright = (b - c) / 6 * (fc + 4 * fe + fb)
        S2 = Sleft + Sright
        
        if depth <= 0 or abs(S2 - S) <= 15 * tol:
            return S2 + (S2 - S) / 15
        
        return (adaptive_aux(f, a, c, tol/2, Sleft, fa, fc, fd, depth-1) +
                adaptive_aux(f, c, b, tol/2, Sright, fc, fb, fe, depth-1))
    
    c = (a + b) / 2
    fa = f(a)
    fb = f(b)
    fc = f(c)
    S = simpson_rule(f, a, b)
    
    return adaptive_aux(f, a, b, tol, S, fa, fb, fc, max_depth)


def gaussian_quadrature(f, a, b, n=5):
    """
    Metode Gaussian Quadrature
    Menggunakan n-point Gauss-Legendre quadrature
    """
    if n == 5:
        # 5-point Gauss-Legendre nodes dan weights
        nodes = np.array([
            -0.9061798459386640,
            -0.5384693101056831,
            0.0,
            0.5384693101056831,
            0.9061798459386640
        ])
        
        weights = np.array([
            0.2369268850561891,
            0.4786286704993665,
            0.5688888888888889,
            0.4786286704993665,
            0.2369268850561891
        ])
    else:
        # Untuk n lain, gunakan numpy (opsional)
        nodes, weights = np.polynomial.legendre.leggauss(n)
    
    # Transformasi dari [-1, 1] ke [a, b]
    result = 0
    for i in range(len(nodes)):
        x = 0.5 * ((b - a) * nodes[i] + (a + b))
        result += weights[i] * f(x)
    
    result *= 0.5 * (b - a)
    return result


def display_integration_results(f, a, b, exact=None, function_name="f(x)"):
    # print("="*65)
    # print("        PERBANDINGAN METODE INTEGRASI NUMERIK")
    # print("="*65)
    print(f"Fungsi     : {function_name}")
    print(f"Interval   : [{a}, {b}]")
    if exact is not None:
        print(f"Hasil eksak: {exact:.10f}\n")
    else:
        print(f"Hasil eksak: (tidak diketahui)\n")

    # --- Trapezoidal untuk n = 4 dan n = 8 ---
    I4 = trapezoidal_rule(f, a, b, 4)
    I8 = trapezoidal_rule(f, a, b, 8)

    # --- Richardson extrapolation dari dua nilai trapezoidal ---
    I_rich = richardson_extrapolation(f, a, b, 4)

    # --- Romberg Integration level 3 ---
    R_table, I_romberg = romberg_integration(f, a, b, max_level=3)

    # --- Adaptive Simpson & Gaussian Quadrature ---
    I_adapt = adaptive_simpson(f, a, b)
    I_gauss = gaussian_quadrature(f, a, b)

    # --- Tampilan hasil ---
    print(f"{'Metode':<30}{'Hasil':>20}{'Error':>15}")
    print("-"*65)

    def print_result(name, val):
        if exact is not None:
            err = abs(val - exact)
            print(f"{name:<30}{val:>20.10f}{err:>15.2e}")
        else:
            print(f"{name:<30}{val:>20.10f}{'':>15}")

    print_result("Trapezoidal (n=4)", I4)
    print_result("Trapezoidal (n=8)", I8)
    print_result("Richardson Extrapolation", I_rich)
    print_result("Romberg Integration (L3)", I_romberg)
    print_result("Adaptive Simpson", I_adapt)
    print_result("Gaussian Quadrature", I_gauss)
    print("-"*65)

    # --- Tabel Romberg ---
    print("\nTabel Romberg:")
    for i in range(len(R_table)):
        row = [f"{val:>12.8f}" if val != 0 else " "*12 for val in R_table[i]]
        print(" ".join(row))
    print("="*65 + "\n")

def main():
    f = lambda x: np.cos(x)
    a, b = 0, np.pi/2
    exact = 1.0

    display_integration_results(f, a, b, exact, "cos(x)")
    
    f = lambda x: x**2
    a, b = 0, 1
    exact = 1.0/3

    display_integration_results(f, a, b, exact, "x^2")
    
        
    
if __name__ == "__main__":
    main()
