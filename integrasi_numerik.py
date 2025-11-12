import numpy as np
import pandas as pd

# ============================================================================
# FUNGSI-FUNGSI INTEGRASI NUMERIK
# ============================================================================

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


def richardson_extrapolation(f, a, b, max_level=5):
    """
    Metode Richardson Extrapolation
    Menggunakan trapezoidal rule sebagai basis dan melakukan ekstrapolasi
    """
    R = np.zeros((max_level, max_level))
    
    # Kolom pertama: hasil trapezoidal rule dengan n = 2^i
    for i in range(max_level):
        n = 2**i
        R[i][0] = trapezoidal_rule(f, a, b, n)
    
    # Richardson extrapolation untuk kolom-kolom berikutnya
    for j in range(1, max_level):
        for i in range(j, max_level):
            R[i][j] = R[i][j-1] + (R[i][j-1] - R[i-1][j-1]) / (4**j - 1)
    
    return R


def romberg_integration(f, a, b, max_level=5):
    """
    Metode Romberg Integration
    Menghasilkan tabel Romberg R[i][j]
    """
    R = np.zeros((max_level, max_level))
    
    # R[0][0] menggunakan trapezoidal rule dengan 1 interval
    R[0][0] = 0.5 * (b - a) * (f(a) + f(b))
    
    for i in range(1, max_level):
        h = (b - a) / (2**i)
        
        # Hitung sum untuk titik-titik baru
        sum_val = 0
        for k in range(1, 2**(i-1) + 1):
            sum_val += f(a + (2*k - 1) * h)
        
        R[i][0] = 0.5 * R[i-1][0] + h * sum_val
        
        # Richardson extrapolation
        for j in range(1, i + 1):
            R[i][j] = R[i][j-1] + (R[i][j-1] - R[i-1][j-1]) / (4**j - 1)
    
    return R


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


# ============================================================================
# FUNGSI-FUNGSI YANG AKAN DIINTEGRALKAN
# ============================================================================

def f1(x):
    """Fungsi 1: cos(x)"""
    return np.cos(x)

def f2(x):
    """Fungsi 2: x^2"""
    return x**2


# ============================================================================
# PROGRAM UTAMA
# ============================================================================

def main():
    print("="*80)
    print("PROGRAM INTEGRASI NUMERIK")
    print("="*80)
    print()
    
    # ========================================================================
    # SOAL 1: Integral cos(x) dari 0 sampai π/2
    # ========================================================================
    print("SOAL 1: I = ∫₀^(π/2) cos(x) dx")
    print("-" * 80)
    
    a1 = 0
    b1 = np.pi / 2
    exact1 = 1.0  # sin(π/2) - sin(0) = 1
    
    print(f"Batas integral: [{a1:.4f}, {b1:.4f}]")
    print(f"Nilai Eksak: {exact1:.15f}")
    print()
    
    # 1. Trapezoidal Rule dengan minimal 3 level
    print("1. TRAPEZOIDAL RULE (minimal 3 level sampai R₂,₂)")
    print("-" * 80)
    
    trap1_R00 = trapezoidal_rule(f1, a1, b1, 1)  # n=1, R₀,₀
    trap1_R11 = trapezoidal_rule(f1, a1, b1, 2)  # n=2, R₁,₁
    trap1_R22 = trapezoidal_rule(f1, a1, b1, 4)  # n=4, R₂,₂
    
    print(f"R₀,₀ (n=1):  {trap1_R00:.15f}  | Error: {abs(trap1_R00 - exact1):.6e}")
    print(f"R₁,₁ (n=2):  {trap1_R11:.15f}  | Error: {abs(trap1_R11 - exact1):.6e}")
    print(f"R₂,₂ (n=4):  {trap1_R22:.15f}  | Error: {abs(trap1_R22 - exact1):.6e}")
    print()
    
    # 2. Richardson Extrapolation
    print("2. RICHARDSON EXTRAPOLATION")
    print("-" * 80)
    
    richardson1 = richardson_extrapolation(f1, a1, b1, max_level=5)
    
    # Tampilkan tabel Richardson
    print("Tabel Richardson Extrapolation:")
    # print(f"{'i\\j':<6}", end="")
    for j in range(5):
        print(f"j={j:<18}", end="")
    print()
    
    for i in range(5):
        print(f"i={i:<4}", end="")
        for j in range(5):
            if j <= i:
                print(f"{richardson1[i][j]:<20.15f}", end="")
            else:
                print(f"{'-':<20}", end="")
        print()
    
    print(f"\nHasil terbaik (R₄,₄): {richardson1[4][4]:.15f}")
    print(f"Error: {abs(richardson1[4][4] - exact1):.6e}")
    print()
    
    # 3. Romberg Integration
    print("3. ROMBERG INTEGRATION (Tabel Romberg)")
    print("-" * 80)
    
    romberg1 = romberg_integration(f1, a1, b1, max_level=5)
    
    # Tampilkan tabel Romberg
    print("Tabel Romberg:")
    # print(f"{'i\\j':<6}", end="")
    for j in range(5):
        print(f"j={j:<18}", end="")
    print()
    
    for i in range(5):
        print(f"i={i:<4}", end="")
        for j in range(5):
            if j <= i:
                print(f"{romberg1[i][j]:<20.15f}", end="")
            else:
                print(f"{'-':<20}", end="")
        print()
    
    print(f"\nHasil terbaik (R₄,₄): {romberg1[4][4]:.15f}")
    print(f"Error: {abs(romberg1[4][4] - exact1):.6e}")
    print()
    
    # 4. Adaptive Integration
    print("4. ADAPTIVE INTEGRATION (Adaptive Simpson)")
    print("-" * 80)
    
    adaptive1 = adaptive_simpson(f1, a1, b1)
    print(f"Hasil: {adaptive1:.15f}")
    print(f"Error: {abs(adaptive1 - exact1):.6e}")
    print()
    
    # 5. Gaussian Quadrature
    print("5. GAUSSIAN QUADRATURE (5-point)")
    print("-" * 80)
    
    gauss1 = gaussian_quadrature(f1, a1, b1, n=5)
    print(f"Hasil: {gauss1:.15f}")
    print(f"Error: {abs(gauss1 - exact1):.6e}")
    print()
    
    # Ringkasan Soal 1
    print("RINGKASAN HASIL SOAL 1:")
    print("-" * 80)
    results1 = pd.DataFrame({
        'Metode': [
            'Trapezoidal R₀,₀',
            'Trapezoidal R₁,₁',
            'Trapezoidal R₂,₂',
            'Richardson R₄,₄',
            'Romberg R₄,₄',
            'Adaptive Simpson',
            'Gaussian Quadrature'
        ],
        'Hasil': [
            trap1_R00,
            trap1_R11,
            trap1_R22,
            richardson1[4][4],
            romberg1[4][4],
            adaptive1,
            gauss1
        ],
        'Error': [
            abs(trap1_R00 - exact1),
            abs(trap1_R11 - exact1),
            abs(trap1_R22 - exact1),
            abs(richardson1[4][4] - exact1),
            abs(romberg1[4][4] - exact1),
            abs(adaptive1 - exact1),
            abs(gauss1 - exact1)
        ]
    })
    print(results1.to_string(index=False))
    print()
    print()
    
    # ========================================================================
    # SOAL 2: Integral x^2 dari 0 sampai 1
    # ========================================================================
    print("="*80)
    print("SOAL 2: I = ∫₀¹ x² dx")
    print("-" * 80)
    
    a2 = 0
    b2 = 1
    exact2 = 1/3  # [x³/3] dari 0 ke 1 = 1/3
    
    print(f"Batas integral: [{a2:.4f}, {b2:.4f}]")
    print(f"Nilai Eksak: {exact2:.15f}")
    print()
    
    # 1. Trapezoidal Rule dengan minimal 3 level
    print("1. TRAPEZOIDAL RULE (minimal 3 level sampai R₂,₂)")
    print("-" * 80)
    
    trap2_R00 = trapezoidal_rule(f2, a2, b2, 1)  # n=1, R₀,₀
    trap2_R11 = trapezoidal_rule(f2, a2, b2, 2)  # n=2, R₁,₁
    trap2_R22 = trapezoidal_rule(f2, a2, b2, 4)  # n=4, R₂,₂
    
    print(f"R₀,₀ (n=1):  {trap2_R00:.15f}  | Error: {abs(trap2_R00 - exact2):.6e}")
    print(f"R₁,₁ (n=2):  {trap2_R11:.15f}  | Error: {abs(trap2_R11 - exact2):.6e}")
    print(f"R₂,₂ (n=4):  {trap2_R22:.15f}  | Error: {abs(trap2_R22 - exact2):.6e}")
    print()
    
    # 2. Richardson Extrapolation
    print("2. RICHARDSON EXTRAPOLATION")
    print("-" * 80)
    
    richardson2 = richardson_extrapolation(f2, a2, b2, max_level=5)
    
    # Tampilkan tabel Richardson
    print("Tabel Richardson Extrapolation:")
    # print(f"{'i\\j':<6}", end="")
    for j in range(5):
        print(f"j={j:<18}", end="")
    print()
    
    for i in range(5):
        print(f"i={i:<4}", end="")
        for j in range(5):
            if j <= i:
                print(f"{richardson2[i][j]:<20.15f}", end="")
            else:
                print(f"{'-':<20}", end="")
        print()
    
    print(f"\nHasil terbaik (R₄,₄): {richardson2[4][4]:.15f}")
    print(f"Error: {abs(richardson2[4][4] - exact2):.6e}")
    print()
    
    # 3. Romberg Integration
    print("3. ROMBERG INTEGRATION (Tabel Romberg)")
    print("-" * 80)
    
    romberg2 = romberg_integration(f2, a2, b2, max_level=5)
    
    # Tampilkan tabel Romberg
    print("Tabel Romberg:")
    # print(f"{'i\\j':<6}", end="")
    for j in range(5):
        print(f"j={j:<18}", end="")
    print()
    
    for i in range(5):
        print(f"i={i:<4}", end="")
        for j in range(5):
            if j <= i:
                print(f"{romberg2[i][j]:<20.15f}", end="")
            else:
                print(f"{'-':<20}", end="")
        print()
    
    print(f"\nHasil terbaik (R₄,₄): {romberg2[4][4]:.15f}")
    print(f"Error: {abs(romberg2[4][4] - exact2):.6e}")
    print()
    
    # 4. Adaptive Integration
    print("4. ADAPTIVE INTEGRATION (Adaptive Simpson)")
    print("-" * 80)
    
    adaptive2 = adaptive_simpson(f2, a2, b2)
    print(f"Hasil: {adaptive2:.15f}")
    print(f"Error: {abs(adaptive2 - exact2):.6e}")
    print()
    
    # 5. Gaussian Quadrature
    print("5. GAUSSIAN QUADRATURE (5-point)")
    print("-" * 80)
    
    gauss2 = gaussian_quadrature(f2, a2, b2, n=5)
    print(f"Hasil: {gauss2:.15f}")
    print(f"Error: {abs(gauss2 - exact2):.6e}")
    print()
    
    # Ringkasan Soal 2
    print("RINGKASAN HASIL SOAL 2:")
    print("-" * 80)
    results2 = pd.DataFrame({
        'Metode': [
            'Trapezoidal R₀,₀',
            'Trapezoidal R₁,₁',
            'Trapezoidal R₂,₂',
            'Richardson R₄,₄',
            'Romberg R₄,₄',
            'Adaptive Simpson',
            'Gaussian Quadrature'
        ],
        'Hasil': [
            trap2_R00,
            trap2_R11,
            trap2_R22,
            richardson2[4][4],
            romberg2[4][4],
            adaptive2,
            gauss2
        ],
        'Error': [
            abs(trap2_R00 - exact2),
            abs(trap2_R11 - exact2),
            abs(trap2_R22 - exact2),
            abs(richardson2[4][4] - exact2),
            abs(romberg2[4][4] - exact2),
            abs(adaptive2 - exact2),
            abs(gauss2 - exact2)
        ]
    })
    print(results2.to_string(index=False))
    print()
    
    print("="*80)
    print("PROGRAM SELESAI")
    print("="*80)


if __name__ == "__main__":
    main()