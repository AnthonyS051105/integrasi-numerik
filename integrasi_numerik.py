import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as mpatches

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
# FUNGSI UNTUK MEMBUAT PDF
# ============================================================================

def create_pdf_report(results1, results2, romberg1, romberg2, exact1, exact2):
    """
    Membuat laporan PDF dengan langkah-langkah matematis
    """
    pdf_filename = "hasil_integrasi_numerik.pdf"
    
    with PdfPages(pdf_filename) as pdf:
        # Halaman 1: Cover dan Pengenalan
        fig = plt.figure(figsize=(8.5, 11))
        fig.suptitle('LAPORAN INTEGRASI NUMERIK', fontsize=20, fontweight='bold', y=0.95)
        
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        intro_text = """
METODE NUMERIK - INTEGRASI NUMERIK
===================================

Program ini menyelesaikan dua integral menggunakan empat metode numerik:
1. Trapezoidal Rule
2. Romberg Integration
3. Adaptive Integration (Simpson)
4. Gaussian Quadrature

SOAL YANG DISELESAIKAN:
-----------------------
Soal 1: ∫₀^(π/2) cos(x) dx
Soal 2: ∫₀¹ x² dx

Setiap metode akan dijelaskan dengan langkah-langkah matematis yang detail.
        """
        
        ax.text(0.1, 0.7, intro_text, fontsize=11, verticalalignment='top', 
                fontfamily='monospace')
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Halaman 2: Teori Trapezoidal Rule
        fig = plt.figure(figsize=(8.5, 11))
        fig.suptitle('METODE 1: TRAPEZOIDAL RULE', fontsize=16, fontweight='bold', y=0.95)
        
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        trap_theory = r"""
TEORI TRAPEZOIDAL RULE
======================

Formula Dasar:
--------------
∫ₐᵇ f(x) dx ≈ h/2 [f(x₀) + 2f(x₁) + 2f(x₂) + ... + 2f(xₙ₋₁) + f(xₙ)]

dimana:
  h = (b - a) / n  (lebar subinterval)
  xᵢ = a + i·h      (titik-titik partisi)
  n = jumlah subinterval

Ide Geometris:
--------------
Metode ini mengaproksimasi area di bawah kurva dengan trapesium.
Setiap subinterval [xᵢ, xᵢ₊₁] diaproksimasi dengan trapesium yang memiliki:
  - Tinggi = h = xᵢ₊₁ - xᵢ
  - Sisi sejajar = f(xᵢ) dan f(xᵢ₊₁)
  - Luas trapesium = h/2 [f(xᵢ) + f(xᵢ₊₁)]

Error Analysis:
--------------
Error = -((b-a)³/12n²) · f''(ξ)  untuk ξ ∈ [a,b]
Error berkurang sebanding dengan 1/n² (orde O(h²))

Hubungan dengan Romberg:
-------------------------
Trapezoidal rule dengan n=2ⁱ menghasilkan R(i,0) dalam tabel Romberg:
  - R(0,0): n = 1 (2⁰)
  - R(1,0): n = 2 (2¹)
  - R(2,0): n = 4 (2²)
  - dst.
        """
        
        ax.text(0.05, 0.85, trap_theory, fontsize=9, verticalalignment='top', 
                fontfamily='monospace', linespacing=1.5)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Halaman 3: Perhitungan Soal 1 - Trapezoidal
        fig = plt.figure(figsize=(8.5, 11))
        fig.suptitle('SOAL 1: ∫₀^(π/2) cos(x) dx - TRAPEZOIDAL RULE', 
                     fontsize=14, fontweight='bold', y=0.95)
        
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        a1 = 0
        b1 = np.pi / 2
        
        # Perhitungan manual untuk R₀,₀
        h00 = (b1 - a1) / 1
        x00 = [a1, b1]
        y00 = [np.cos(x) for x in x00]
        trap_R00 = h00 / 2 * (y00[0] + y00[1])
        
        # Perhitungan manual untuk R₁,₁
        h11 = (b1 - a1) / 2
        x11 = [a1, a1 + h11, b1]
        y11 = [np.cos(x) for x in x11]
        trap_R11 = h11 / 2 * (y11[0] + 2*y11[1] + y11[2])
        
        trap_calc = f"""
LANGKAH-LANGKAH PERHITUNGAN:
============================

Diketahui:
  f(x) = cos(x)
  a = 0, b = π/2 ≈ {b1:.6f}
  Nilai eksak = sin(π/2) - sin(0) = 1.0

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LEVEL 0: R₀,₀ (n=1, satu interval)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

h = (π/2 - 0)/1 = {h00:.6f}

Titik evaluasi:
  x₀ = {x00[0]:.6f}, f(x₀) = cos({x00[0]:.4f}) = {y00[0]:.6f}
  x₁ = {x00[1]:.6f}, f(x₁) = cos({x00[1]:.4f}) = {y00[1]:.6f}

R₀,₀ = h/2 [f(x₀) + f(x₁)]
     = {h00:.6f}/2 × [{y00[0]:.6f} + {y00[1]:.6f}]
     = {h00/2:.6f} × {y00[0]+y00[1]:.6f}
     = {trap_R00:.15f}

Error = |{trap_R00:.6f} - 1.0| = {abs(trap_R00 - 1.0):.6e}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LEVEL 1: R₁,₁ (n=2, dua interval)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

h = (π/2 - 0)/2 = {h11:.6f}

Titik evaluasi:
  x₀ = {x11[0]:.6f}, f(x₀) = cos({x11[0]:.4f}) = {y11[0]:.6f}
  x₁ = {x11[1]:.6f}, f(x₁) = cos({x11[1]:.4f}) = {y11[1]:.6f}
  x₂ = {x11[2]:.6f}, f(x₂) = cos({x11[2]:.4f}) = {y11[2]:.6f}

R₁,₁ = h/2 [f(x₀) + 2f(x₁) + f(x₂)]
     = {h11:.6f}/2 × [{y11[0]:.6f} + 2×{y11[1]:.6f} + {y11[2]:.6f}]
     = {h11/2:.6f} × {y11[0] + 2*y11[1] + y11[2]:.6f}
     = {trap_R11:.15f}

Error = |{trap_R11:.6f} - 1.0| = {abs(trap_R11 - 1.0):.6e}
        """
        
        ax.text(0.05, 0.88, trap_calc, fontsize=8, verticalalignment='top', 
                fontfamily='monospace', linespacing=1.3)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Halaman 4: Teori Romberg Integration
        fig = plt.figure(figsize=(8.5, 11))
        fig.suptitle('METODE 2: ROMBERG INTEGRATION', fontsize=16, fontweight='bold', y=0.95)
        
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        romberg_theory = r"""
TEORI ROMBERG INTEGRATION
=========================

Romberg integration adalah metode ekstrapolasi Richardson yang diterapkan
pada Trapezoidal Rule untuk meningkatkan akurasi.

Formula Rekursif:
-----------------
R(i,0) = Trapezoidal rule dengan n = 2ⁱ

R(i,j) = R(i,j-1) + [R(i,j-1) - R(i-1,j-1)] / (4ʲ - 1)

Struktur Tabel Romberg:
-----------------------
       j=0         j=1         j=2         j=3         j=4
i=0  R(0,0)
i=1  R(1,0)     R(1,1)
i=2  R(2,0)     R(2,1)     R(2,2)
i=3  R(3,0)     R(3,1)     R(3,2)     R(3,3)
i=4  R(4,0)     R(4,1)     R(4,2)     R(4,3)     R(4,4)

Penjelasan Kolom:
-----------------
• j=0: Hasil Trapezoidal Rule (orde h²)
• j=1: Ekstrapolasi pertama → Simpson's Rule (orde h⁴)
• j=2: Ekstrapolasi kedua → Boole's Rule (orde h⁶)
• j=3: Ekstrapolasi ketiga (orde h⁸)
• j=4: Ekstrapolasi keempat (orde h¹⁰)

Kelebihan:
----------
1. Konvergensi sangat cepat (eksponensial)
2. Memberikan estimasi error otomatis
3. Akurasi tinggi dengan evaluasi fungsi yang relatif sedikit

Perhitungan R(i,0) untuk i > 0:
-------------------------------
Menggunakan formula rekursif untuk menghindari perhitungan ulang:
h = (b - a) / 2ⁱ
R(i,0) = R(i-1,0)/2 + h × Σ f(a + (2k-1)h)  untuk k=1 hingga 2^(i-1)
        """
        
        ax.text(0.05, 0.85, romberg_theory, fontsize=9, verticalalignment='top', 
                fontfamily='monospace', linespacing=1.5)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Halaman 5: Romberg Soal 1 - Perhitungan Detail
        fig = plt.figure(figsize=(8.5, 11))
        fig.suptitle('SOAL 1: ROMBERG INTEGRATION - PERHITUNGAN', 
                     fontsize=14, fontweight='bold', y=0.95)
        
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        R = romberg1
        romberg_calc = f"""
PERHITUNGAN TABEL ROMBERG untuk ∫₀^(π/2) cos(x) dx
==================================================

KOLOM 0: Trapezoidal Rule dengan n = 2ⁱ
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

R(0,0) = (π/2)/2 × [f(0) + f(π/2)]
       = {(b1-a1)/2:.6f} × [1.0 + 0.0]
       = {R[0][0]:.15f}

R(1,0) menggunakan formula rekursif:
h = π/4, titik baru: π/4
R(1,0) = R(0,0)/2 + h × f(π/4)
       = {R[0][0]/2:.6f} + {b1/4:.6f} × {np.cos(b1/2):.6f}
       = {R[1][0]:.15f}

R(2,0), R(3,0), R(4,0) dihitung serupa dengan n=4, 8, 16

EKSTRAPOLASI (Kolom j≥1):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

R(i,j) = R(i,j-1) + [R(i,j-1) - R(i-1,j-1)] / (4ʲ - 1)

Contoh: R(1,1) (Simpson's Rule)
R(1,1) = R(1,0) + [R(1,0) - R(0,0)] / (4¹ - 1)
       = {R[1][0]:.10f} + [{R[1][0]:.6f} - {R[0][0]:.6f}] / 3
       = {R[1][0]:.10f} + {(R[1][0]-R[0][0])/3:.10f}
       = {R[1][1]:.15f}

Contoh: R(2,1)
R(2,1) = R(2,0) + [R(2,0) - R(1,0)] / 3
       = {R[2][0]:.10f} + {(R[2][0]-R[1][0])/3:.10f}
       = {R[2][1]:.15f}

Contoh: R(2,2)
R(2,2) = R(2,1) + [R(2,1) - R(1,1)] / (4² - 1)
       = {R[2][1]:.10f} + [{R[2][1]:.6f} - {R[1][1]:.6f}] / 15
       = {R[2][1]:.10f} + {(R[2][1]-R[1][1])/15:.10f}
       = {R[2][2]:.15f}

HASIL TERBAIK: R(4,4) = {R[4][4]:.15f}
Error = {abs(R[4][4] - exact1):.6e}
        """
        
        ax.text(0.05, 0.88, romberg_calc, fontsize=7.5, verticalalignment='top', 
                fontfamily='monospace', linespacing=1.3)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Halaman 6: Tabel Romberg Lengkap Soal 1
        fig = plt.figure(figsize=(8.5, 11))
        fig.suptitle('SOAL 1: TABEL ROMBERG LENGKAP', 
                     fontsize=14, fontweight='bold', y=0.95)
        
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        # Buat tabel Romberg
        table_text = "TABEL ROMBERG LENGKAP:\n"
        table_text += "="*90 + "\n\n"
        table_text += f"{'i\\j':<6}"
        for j in range(5):
            table_text += f"{'j='+str(j):<18}"
        table_text += "\n" + "-"*90 + "\n"
        
        for i in range(5):
            table_text += f"i={i:<4}"
            for j in range(5):
                if j <= i:
                    table_text += f"{R[i][j]:<18.12f}"
                else:
                    table_text += f"{'-':<18}"
            table_text += "\n"
        
        table_text += "\n\nINTERPRETASI:\n"
        table_text += "-" * 90 + "\n"
        table_text += f"• Diagonal utama menunjukkan hasil dengan akurasi meningkat\n"
        table_text += f"• R(4,4) memiliki akurasi tertinggi dengan error {abs(R[4][4]-exact1):.2e}\n"
        table_text += f"• Setiap ekstrapolasi meningkatkan orde akurasi sebesar O(h²)\n"
        
        ax.text(0.05, 0.85, table_text, fontsize=9, verticalalignment='top', 
                fontfamily='monospace', linespacing=1.4)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Halaman 7: Teori Adaptive Integration
        fig = plt.figure(figsize=(8.5, 11))
        fig.suptitle('METODE 3: ADAPTIVE INTEGRATION', fontsize=16, fontweight='bold', y=0.95)
        
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        adaptive_theory = r"""
TEORI ADAPTIVE INTEGRATION
===========================

Adaptive Integration adalah metode yang secara otomatis menyesuaikan ukuran
subinterval berdasarkan perilaku fungsi untuk mencapai toleransi error tertentu.

Formula Simpson's Rule:
-----------------------
Untuk interval [a,b] dengan midpoint c = (a+b)/2:

S = (b-a)/6 × [f(a) + 4f(c) + f(b)]

Algoritma Adaptive:
-------------------
1. Hitung S untuk seluruh interval [a,b]
2. Bagi interval menjadi dua: [a,c] dan [c,b]
3. Hitung S_left dan S_right untuk masing-masing subinterval
4. Hitung S2 = S_left + S_right
5. Jika |S2 - S| ≤ 15·tol, terima S2 + (S2-S)/15
6. Jika tidak, rekursi pada [a,c] dan [c,b] dengan tol/2

Estimasi Error:
--------------
Error ≈ (S2 - S) / 15

Ini berdasarkan analisis error Simpson's rule:
  Error_Simpson ∝ h⁵

Ketika h dibagi dua, error berkurang dengan faktor 2⁵ = 32
Sehingga: Error_new ≈ Error_old / 16
Dan: Error ≈ (S2 - S) / (16-1) = (S2 - S) / 15

Kelebihan:
----------
1. Efisien: fokus pada daerah yang sulit
2. Akurasi terkontrol dengan toleransi
3. Otomatis menyesuaikan dengan kompleksitas fungsi
4. Menghindari over-computation pada daerah yang smooth

Parameter:
----------
• tol: toleransi error yang diinginkan (default: 1e-10)
• max_depth: kedalaman rekursi maksimum (default: 50)
        """
        
        ax.text(0.05, 0.85, adaptive_theory, fontsize=9, verticalalignment='top', 
                fontfamily='monospace', linespacing=1.5)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Halaman 8: Teori Gaussian Quadrature
        fig = plt.figure(figsize=(8.5, 11))
        fig.suptitle('METODE 4: GAUSSIAN QUADRATURE', fontsize=16, fontweight='bold', y=0.95)
        
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        gauss_theory = r"""
TEORI GAUSSIAN QUADRATURE
=========================

Gaussian Quadrature adalah metode integrasi yang menggunakan titik-titik
sampling optimal (Gauss points) untuk mencapai akurasi maksimal.

Formula Dasar (Gauss-Legendre):
--------------------------------
∫₋₁¹ f(x) dx ≈ Σᵢ wᵢ · f(xᵢ)

dimana:
  xᵢ: Gauss points (akar polinomial Legendre)
  wᵢ: bobot (weights) yang sesuai

5-Point Gauss-Legendre:
-----------------------
Nodes (xᵢ):                        Weights (wᵢ):
  x₁ = -0.9061798459386640         w₁ = 0.2369268850561891
  x₂ = -0.5384693101056831         w₂ = 0.4786286704993665
  x₃ =  0.0                        w₃ = 0.5688888888888889
  x₄ =  0.5384693101056831         w₄ = 0.4786286704993665
  x₅ =  0.9061798459386640         w₅ = 0.2369268850561891

Catatan: Weights simetris (w₁=w₅, w₂=w₄)

Transformasi ke Interval [a,b]:
-------------------------------
∫ₐᵇ f(x) dx = (b-a)/2 · ∫₋₁¹ f((b-a)t/2 + (a+b)/2) dt

Langkah-langkah:
1. Transformasi xᵢ dari [-1,1] ke [a,b]:
   x'ᵢ = (b-a)/2 · xᵢ + (a+b)/2

2. Hitung weighted sum:
   Σᵢ wᵢ · f(x'ᵢ)

3. Kalikan dengan faktor skala:
   (b-a)/2 · Σᵢ wᵢ · f(x'ᵢ)

Akurasi:
--------
n-point Gauss quadrature exact untuk polinomial derajat ≤ 2n-1
5-point Gauss quadrature exact untuk polinomial derajat ≤ 9

Kelebihan:
----------
1. Akurasi sangat tinggi dengan evaluasi fungsi minimal
2. Optimal untuk fungsi smooth
3. Tidak memerlukan turunan fungsi
4. Error ∝ f⁽²ⁿ⁾(ξ) dengan orde yang sangat tinggi
        """
        
        ax.text(0.05, 0.85, gauss_theory, fontsize=8.5, verticalalignment='top', 
                fontfamily='monospace', linespacing=1.5)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Halaman 9: Gaussian Quadrature Soal 1 - Detail
        fig = plt.figure(figsize=(8.5, 11))
        fig.suptitle('SOAL 1: GAUSSIAN QUADRATURE - PERHITUNGAN', 
                     fontsize=14, fontweight='bold', y=0.95)
        
        ax = fig.add_subplot(111)
        ax.axis('off')
        
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
        
        # Transformasi nodes
        transformed_nodes = [0.5 * ((b1 - a1) * node + (a1 + b1)) for node in nodes]
        f_values = [np.cos(x) for x in transformed_nodes]
        
        gauss_calc = f"""
PERHITUNGAN GAUSSIAN QUADRATURE untuk ∫₀^(π/2) cos(x) dx
========================================================

TRANSFORMASI dari [-1,1] ke [0, π/2]:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Formula: x'ᵢ = (b-a)/2 · xᵢ + (a+b)/2
       = {(b1-a1)/2:.8f} · xᵢ + {(a1+b1)/2:.8f}

EVALUASI TITIK-TITIK:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

i   Node (xᵢ)           x'ᵢ (transformed)  f(x'ᵢ)=cos(x'ᵢ)   Weight (wᵢ)
─────────────────────────────────────────────────────────────────────────
1  {nodes[0]:>18.15f}  {transformed_nodes[0]:>17.15f}  {f_values[0]:>15.12f}  {weights[0]:.15f}
2  {nodes[1]:>18.15f}  {transformed_nodes[1]:>17.15f}  {f_values[1]:>15.12f}  {weights[1]:.15f}
3  {nodes[2]:>18.15f}  {transformed_nodes[2]:>17.15f}  {f_values[2]:>15.12f}  {weights[2]:.15f}
4  {nodes[3]:>18.15f}  {transformed_nodes[3]:>17.15f}  {f_values[3]:>15.12f}  {weights[3]:.15f}
5  {nodes[4]:>18.15f}  {transformed_nodes[4]:>17.15f}  {f_values[4]:>15.12f}  {weights[4]:.15f}

PERHITUNGAN WEIGHTED SUM:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Σ wᵢ · f(x'ᵢ) = w₁·f(x'₁) + w₂·f(x'₂) + w₃·f(x'₃) + w₄·f(x'₄) + w₅·f(x'₅)

             = {weights[0]:.6f} × {f_values[0]:.6f}
             + {weights[1]:.6f} × {f_values[1]:.6f}
             + {weights[2]:.6f} × {f_values[2]:.6f}
             + {weights[3]:.6f} × {f_values[3]:.6f}
             + {weights[4]:.6f} × {f_values[4]:.6f}

             = {sum(weights[i]*f_values[i] for i in range(5)):.15f}

HASIL AKHIR:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

I ≈ (b-a)/2 × Σ wᵢ · f(x'ᵢ)
  = {(b1-a1)/2:.8f} × {sum(weights[i]*f_values[i] for i in range(5)):.15f}
  = {gaussian_quadrature(f1, a1, b1, n=5):.15f}

Error = {abs(gaussian_quadrature(f1, a1, b1, n=5) - exact1):.6e}

CATATAN:
Error sangat kecil karena cos(x) sangat smooth dan dapat diaproksimasi
dengan baik oleh polinomial derajat tinggi.
        """
        
        ax.text(0.05, 0.88, gauss_calc, fontsize=7, verticalalignment='top', 
                fontfamily='monospace', linespacing=1.25)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Halaman 10: Ringkasan Soal 1
        fig = plt.figure(figsize=(8.5, 11))
        fig.suptitle('SOAL 1: RINGKASAN HASIL', 
                     fontsize=14, fontweight='bold', y=0.95)
        
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        summary1 = f"""
RINGKASAN HASIL SOAL 1: ∫₀^(π/2) cos(x) dx
==========================================

Nilai Eksak: {exact1:.15f}

PERBANDINGAN SEMUA METODE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"""
        
        # Tambahkan tabel hasil
        for idx, row in results1.iterrows():
            summary1 += f"{row['Metode']:<25} {row['Hasil']:.15f}  Error: {row['Error']:.6e}\n"
        
        summary1 += f"""

ANALISIS PERFORMA:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. TRAPEZOIDAL RULE:
   • R₀,₀ (n=1): Error terbesar {results1.iloc[0]['Error']:.2e}
   • R₂,₂ (n=4): Error {results1.iloc[2]['Error']:.2e}, improvement ~{results1.iloc[0]['Error']/results1.iloc[2]['Error']:.1f}x
   • Pola: Error berkurang ~4x setiap level (sesuai teori O(h²))

2. ROMBERG INTEGRATION:
   • Error: {results1.iloc[3]['Error']:.2e}
   • Metode tercepat konvergen dengan akurasi sangat tinggi
   • Hanya perlu 2⁴=16 evaluasi fungsi untuk hasil luar biasa akurat

3. ADAPTIVE SIMPSON:
   • Error: {results1.iloc[4]['Error']:.2e}
   • Otomatis menyesuaikan dengan kompleksitas fungsi
   • Cocok untuk fungsi dengan perilaku tidak teratur

4. GAUSSIAN QUADRATURE:
   • Error: {results1.iloc[5]['Error']:.2e}
   • Akurasi tertinggi dengan hanya 5 evaluasi fungsi!
   • Optimal untuk fungsi smooth seperti cos(x)
   • Exact untuk polinomial derajat ≤9

KESIMPULAN:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Untuk integral cos(x):
• Gaussian Quadrature memberikan hasil terbaik (error {results1.iloc[5]['Error']:.2e})
• Romberg juga sangat baik (error {results1.iloc[3]['Error']:.2e})
• Semua metode konvergen ke nilai eksak dengan baik
        """
        
        ax.text(0.05, 0.88, summary1, fontsize=8.5, verticalalignment='top', 
                fontfamily='monospace', linespacing=1.4)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Halaman 11-15: Ulangi untuk Soal 2
        # Halaman 11: Perhitungan Soal 2 - Trapezoidal
        fig = plt.figure(figsize=(8.5, 11))
        fig.suptitle('SOAL 2: ∫₀¹ x² dx - TRAPEZOIDAL RULE', 
                     fontsize=14, fontweight='bold', y=0.95)
        
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        a2 = 0
        b2 = 1
        
        # Perhitungan manual untuk R₀,₀
        h00_2 = (b2 - a2) / 1
        x00_2 = [a2, b2]
        y00_2 = [x**2 for x in x00_2]
        trap2_R00 = h00_2 / 2 * (y00_2[0] + y00_2[1])
        
        # Perhitungan manual untuk R₁,₁
        h11_2 = (b2 - a2) / 2
        x11_2 = [a2, a2 + h11_2, b2]
        y11_2 = [x**2 for x in x11_2]
        trap2_R11 = h11_2 / 2 * (y11_2[0] + 2*y11_2[1] + y11_2[2])
        
        trap2_calc = f"""
LANGKAH-LANGKAH PERHITUNGAN:
============================

Diketahui:
  f(x) = x²
  a = 0, b = 1
  Nilai eksak = ∫₀¹ x² dx = [x³/3]₀¹ = 1/3 = {exact2:.15f}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LEVEL 0: R₀,₀ (n=1, satu interval)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

h = (1 - 0)/1 = {h00_2:.6f}

Titik evaluasi:
  x₀ = {x00_2[0]:.6f}, f(x₀) = ({x00_2[0]:.4f})² = {y00_2[0]:.6f}
  x₁ = {x00_2[1]:.6f}, f(x₁) = ({x00_2[1]:.4f})² = {y00_2[1]:.6f}

R₀,₀ = h/2 [f(x₀) + f(x₁)]
     = {h00_2:.6f}/2 × [{y00_2[0]:.6f} + {y00_2[1]:.6f}]
     = {h00_2/2:.6f} × {y00_2[0]+y00_2[1]:.6f}
     = {trap2_R00:.15f}

Error = |{trap2_R00:.6f} - {exact2:.6f}| = {abs(trap2_R00 - exact2):.6e}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LEVEL 1: R₁,₁ (n=2, dua interval)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

h = (1 - 0)/2 = {h11_2:.6f}

Titik evaluasi:
  x₀ = {x11_2[0]:.6f}, f(x₀) = ({x11_2[0]:.4f})² = {y11_2[0]:.6f}
  x₁ = {x11_2[1]:.6f}, f(x₁) = ({x11_2[1]:.4f})² = {y11_2[1]:.6f}
  x₂ = {x11_2[2]:.6f}, f(x₂) = ({x11_2[2]:.4f})² = {y11_2[2]:.6f}

R₁,₁ = h/2 [f(x₀) + 2f(x₁) + f(x₂)]
     = {h11_2:.6f}/2 × [{y11_2[0]:.6f} + 2×{y11_2[1]:.6f} + {y11_2[2]:.6f}]
     = {h11_2/2:.6f} × {y11_2[0] + 2*y11_2[1] + y11_2[2]:.6f}
     = {trap2_R11:.15f}

Error = |{trap2_R11:.6f} - {exact2:.6f}| = {abs(trap2_R11 - exact2):.6e}

LEVEL 2: R₂,₂ (n=4)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Dengan 4 interval (h=0.25), kita evaluasi di x=0, 0.25, 0.5, 0.75, 1
Hasil: {trapezoidal_rule(f2, a2, b2, 4):.15f}
Error: {abs(trapezoidal_rule(f2, a2, b2, 4) - exact2):.6e}
        """
        
        ax.text(0.05, 0.88, trap2_calc, fontsize=8, verticalalignment='top', 
                fontfamily='monospace', linespacing=1.3)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Halaman 12: Tabel Romberg Soal 2
        fig = plt.figure(figsize=(8.5, 11))
        fig.suptitle('SOAL 2: ROMBERG INTEGRATION', 
                     fontsize=14, fontweight='bold', y=0.95)
        
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        R2 = romberg2
        
        # Buat tabel Romberg
        table2_text = f"""
TABEL ROMBERG untuk ∫₀¹ x² dx
==============================

Nilai Eksak: {exact2:.15f}

TABEL LENGKAP:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"""
        table2_text += f"{'i\\j':<6}"
        for j in range(5):
            table2_text += f"{'j='+str(j):<18}"
        table2_text += "\n" + "-"*90 + "\n"
        
        for i in range(5):
            table2_text += f"i={i:<4}"
            for j in range(5):
                if j <= i:
                    table2_text += f"{R2[i][j]:<18.12f}"
                else:
                    table2_text += f"{'-':<18}"
            table2_text += "\n"
        
        table2_text += f"""

CONTOH PERHITUNGAN EKSTRAPOLASI:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

R(1,1) = R(1,0) + [R(1,0) - R(0,0)] / 3
       = {R2[1][0]:.10f} + [{R2[1][0]:.10f} - {R2[0][0]:.10f}] / 3
       = {R2[1][0]:.10f} + {(R2[1][0]-R2[0][0])/3:.10f}
       = {R2[1][1]:.15f}

R(2,2) = R(2,1) + [R(2,1) - R(1,1)] / 15
       = {R2[2][1]:.10f} + {(R2[2][1]-R2[1][1])/15:.10f}
       = {R2[2][2]:.15f}

R(3,3) = R(3,2) + [R(3,2) - R(2,2)] / 63
       = {R2[3][2]:.10f} + {(R2[3][2]-R2[2][2])/63:.10f}
       = {R2[3][3]:.15f}

R(4,4) = R(4,3) + [R(4,3) - R(3,3)] / 255
       = {R2[4][3]:.10f} + {(R2[4][3]-R2[3][3])/255:.10f}
       = {R2[4][4]:.15f}

HASIL AKHIR: R(4,4) = {R2[4][4]:.15f}
Error = {abs(R2[4][4] - exact2):.6e}

OBSERVASI:
Untuk fungsi polinomial x², konvergensi sangat cepat karena
Simpson's Rule (R(1,1)) sudah exact untuk polinomial derajat ≤3!
        """
        
        ax.text(0.05, 0.88, table2_text, fontsize=7.5, verticalalignment='top', 
                fontfamily='monospace', linespacing=1.3)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Halaman 13: Gaussian Quadrature Soal 2
        fig = plt.figure(figsize=(8.5, 11))
        fig.suptitle('SOAL 2: GAUSSIAN QUADRATURE - PERHITUNGAN', 
                     fontsize=14, fontweight='bold', y=0.95)
        
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        # Transformasi nodes untuk soal 2
        transformed_nodes_2 = [0.5 * ((b2 - a2) * node + (a2 + b2)) for node in nodes]
        f2_values = [x**2 for x in transformed_nodes_2]
        
        gauss2_calc = f"""
PERHITUNGAN GAUSSIAN QUADRATURE untuk ∫₀¹ x² dx
===============================================

TRANSFORMASI dari [-1,1] ke [0, 1]:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Formula: x'ᵢ = (b-a)/2 · xᵢ + (a+b)/2
       = {(b2-a2)/2:.8f} · xᵢ + {(a2+b2)/2:.8f}
       = 0.5 · xᵢ + 0.5

EVALUASI TITIK-TITIK:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

i   Node (xᵢ)           x'ᵢ (transformed)  f(x'ᵢ)=(x'ᵢ)²    Weight (wᵢ)
────────────────────────────────────────────────────────────────────────
1  {nodes[0]:>18.15f}  {transformed_nodes_2[0]:>17.15f}  {f2_values[0]:>15.12f}  {weights[0]:.15f}
2  {nodes[1]:>18.15f}  {transformed_nodes_2[1]:>17.15f}  {f2_values[1]:>15.12f}  {weights[1]:.15f}
3  {nodes[2]:>18.15f}  {transformed_nodes_2[2]:>17.15f}  {f2_values[2]:>15.12f}  {weights[2]:.15f}
4  {nodes[3]:>18.15f}  {transformed_nodes_2[3]:>17.15f}  {f2_values[3]:>15.12f}  {weights[3]:.15f}
5  {nodes[4]:>18.15f}  {transformed_nodes_2[4]:>17.15f}  {f2_values[4]:>15.12f}  {weights[4]:.15f}

PERHITUNGAN WEIGHTED SUM:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Σ wᵢ · f(x'ᵢ) = {weights[0]:.6f} × {f2_values[0]:.10f}
             + {weights[1]:.6f} × {f2_values[1]:.10f}
             + {weights[2]:.6f} × {f2_values[2]:.10f}
             + {weights[3]:.6f} × {f2_values[3]:.10f}
             + {weights[4]:.6f} × {f2_values[4]:.10f}

             = {weights[0]*f2_values[0]:.12f}
             + {weights[1]*f2_values[1]:.12f}
             + {weights[2]*f2_values[2]:.12f}
             + {weights[3]*f2_values[3]:.12f}
             + {weights[4]*f2_values[4]:.12f}

             = {sum(weights[i]*f2_values[i] for i in range(5)):.15f}

HASIL AKHIR:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

I ≈ (b-a)/2 × Σ wᵢ · f(x'ᵢ)
  = {(b2-a2)/2:.8f} × {sum(weights[i]*f2_values[i] for i in range(5)):.15f}
  = {gaussian_quadrature(f2, a2, b2, n=5):.15f}

Nilai Eksak: {exact2:.15f}
Error: {abs(gaussian_quadrature(f2, a2, b2, n=5) - exact2):.6e}

CATATAN PENTING:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Karena f(x)=x² adalah polinomial derajat 2, dan 5-point Gauss quadrature
exact untuk polinomial derajat ≤ 9, hasil ini seharusnya exact.
Error yang muncul (~{abs(gaussian_quadrature(f2, a2, b2, n=5) - exact2):.2e}) hanya dari floating-point precision.
        """
        
        ax.text(0.05, 0.88, gauss2_calc, fontsize=7.5, verticalalignment='top', 
                fontfamily='monospace', linespacing=1.25)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Halaman 14: Ringkasan Soal 2
        fig = plt.figure(figsize=(8.5, 11))
        fig.suptitle('SOAL 2: RINGKASAN HASIL', 
                     fontsize=14, fontweight='bold', y=0.95)
        
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        summary2 = f"""
RINGKASAN HASIL SOAL 2: ∫₀¹ x² dx
==================================

Nilai Eksak: {exact2:.15f}

PERBANDINGAN SEMUA METODE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"""
        
        # Tambahkan tabel hasil
        for idx, row in results2.iterrows():
            summary2 += f"{row['Metode']:<25} {row['Hasil']:.15f}  Error: {row['Error']:.6e}\n"
        
        summary2 += f"""

ANALISIS PERFORMA:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. TRAPEZOIDAL RULE:
   • R₀,₀ (n=1): Error {results2.iloc[0]['Error']:.2e}
   • R₂,₂ (n=4): Error {results2.iloc[2]['Error']:.2e}
   • Konvergensi sesuai teori O(h²)

2. ROMBERG INTEGRATION:
   • Error: {results2.iloc[3]['Error']:.2e}
   • Konvergensi sangat cepat karena f(x)=x² adalah polinomial
   • Simpson's Rule (R(1,1)) sudah memberikan akurasi tinggi

3. ADAPTIVE SIMPSON:
   • Error: {results2.iloc[4]['Error']:.2e}
   • Untuk fungsi smooth seperti x², tidak banyak adaptasi diperlukan
   • Akurasi sangat baik dengan minimal subdivision

4. GAUSSIAN QUADRATURE:
   • Error: {results2.iloc[5]['Error']:.2e}
   • Hampir exact! (error hanya floating-point precision)
   • 5-point exact untuk polinomial derajat ≤9
   • x² (derajat 2) → hasil sempurna

KARAKTERISTIK KHUSUS UNTUK POLINOMIAL:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Untuk f(x)=x²:
• Simpson's Rule exact untuk polinomial derajat ≤3 → sangat akurat
• Romberg R(1,1) = Simpson's Rule → konvergensi sangat cepat
• Gaussian Quadrature exact untuk polinomial derajat ≤9 → hasil perfect
• Semua metode memberikan hasil yang sangat baik

KESIMPULAN:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Gaussian Quadrature optimal untuk integral polinomial atau fungsi smooth.
Dengan hanya 5 evaluasi fungsi, error mencapai machine precision!
        """
        
        ax.text(0.05, 0.88, summary2, fontsize=8.5, verticalalignment='top', 
                fontfamily='monospace', linespacing=1.4)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Halaman 15: Kesimpulan Umum
        fig = plt.figure(figsize=(8.5, 11))
        fig.suptitle('KESIMPULAN UMUM DAN REKOMENDASI', 
                     fontsize=16, fontweight='bold', y=0.95)
        
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        conclusion = """
KESIMPULAN UMUM
===============

PERBANDINGAN METODE INTEGRASI NUMERIK
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. TRAPEZOIDAL RULE
   Kelebihan:
   • Sederhana dan mudah diimplementasikan
   • Cocok untuk pemahaman dasar integrasi numerik
   • Stabil secara numerik
   
   Kekurangan:
   • Konvergensi lambat (O(h²))
   • Memerlukan banyak evaluasi fungsi untuk akurasi tinggi
   
   Rekomendasi: Untuk pembelajaran atau quick estimate

2. ROMBERG INTEGRATION
   Kelebihan:
   • Konvergensi sangat cepat (eksponensial)
   • Efisien: akurasi tinggi dengan evaluasi minimal
   • Memberikan sequence of approximations
   • Built-in error estimation
   
   Kekurangan:
   • Memerlukan interval terstruktur (2ⁿ subdivisions)
   • Tidak adaptive
   
   Rekomendasi: Pilihan terbaik untuk fungsi smooth pada interval fixed

3. ADAPTIVE INTEGRATION
   Kelebihan:
   • Otomatis menyesuaikan dengan kompleksitas fungsi
   • Efisien untuk fungsi dengan variasi lokal
   • Kontrol error yang baik
   • Cocok untuk fungsi discontinuous/singular
   
   Kekurangan:
   • Overhead untuk fungsi yang sangat smooth
   • Implementasi lebih kompleks
   
   Rekomendasi: Terbaik untuk fungsi dengan perilaku tidak teratur

4. GAUSSIAN QUADRATURE
   Kelebihan:
   • Akurasi tertinggi per evaluasi fungsi
   • Optimal untuk fungsi smooth
   • n-point exact untuk polinomial derajat ≤2n-1
   • Sangat efisien
   
   Kekurangan:
   • Fixed points (tidak adaptive)
   • Kurang efisien untuk fungsi oscillatory/singular
   • Memerlukan transformasi interval
   
   Rekomendasi: Pilihan terbaik untuk fungsi smooth dan well-behaved

PANDUAN PEMILIHAN METODE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

┌─────────────────────────────┬────────────────────────────────┐
│ KONDISI                     │ METODE YANG DISARANKAN         │
├─────────────────────────────┼────────────────────────────────┤
│ Fungsi smooth & polynomial  │ Gaussian Quadrature            │
│ Fungsi smooth & general     │ Romberg Integration            │
│ Fungsi dengan discontinuity │ Adaptive Integration           │
│ Fungsi oscillatory          │ Adaptive Integration           │
│ Quick estimate              │ Trapezoidal Rule               │
│ High precision required     │ Romberg or Gaussian            │
│ Unknown function behavior   │ Adaptive Integration           │
└─────────────────────────────┴────────────────────────────────┘

HASIL DARI KEDUA SOAL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Soal 1 (cos(x)): Gaussian Quadrature terbaik
Soal 2 (x²):     Gaussian Quadrature hampir exact

Kedua kasus menunjukkan superioritas Gaussian Quadrature untuk
fungsi smooth dan well-behaved.
        """
        
        ax.text(0.05, 0.88, conclusion, fontsize=9, verticalalignment='top', 
                fontfamily='monospace', linespacing=1.5)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    print(f"✓ PDF berhasil dibuat: {pdf_filename}")
    return pdf_filename


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
    
    # 2. Romberg Integration
    print("2. ROMBERG INTEGRATION (Tabel Romberg)")
    print("-" * 80)
    
    romberg1 = romberg_integration(f1, a1, b1, max_level=5)
    
    # Tampilkan tabel Romberg
    print("Tabel Romberg:")
    print(f"{'i\\j':<6}", end="")
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
    
    # 3. Adaptive Integration
    print("3. ADAPTIVE INTEGRATION (Adaptive Simpson)")
    print("-" * 80)
    
    adaptive1 = adaptive_simpson(f1, a1, b1)
    print(f"Hasil: {adaptive1:.15f}")
    print(f"Error: {abs(adaptive1 - exact1):.6e}")
    print()
    
    # 4. Gaussian Quadrature
    print("4. GAUSSIAN QUADRATURE (5-point)")
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
            'Romberg R₄,₄',
            'Adaptive Simpson',
            'Gaussian Quadrature'
        ],
        'Hasil': [
            trap1_R00,
            trap1_R11,
            trap1_R22,
            romberg1[4][4],
            adaptive1,
            gauss1
        ],
        'Error': [
            abs(trap1_R00 - exact1),
            abs(trap1_R11 - exact1),
            abs(trap1_R22 - exact1),
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
    
    # 2. Romberg Integration
    print("2. ROMBERG INTEGRATION (Tabel Romberg)")
    print("-" * 80)
    
    romberg2 = romberg_integration(f2, a2, b2, max_level=5)
    
    # Tampilkan tabel Romberg
    print("Tabel Romberg:")
    print(f"{'i\\j':<6}", end="")
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
    
    # 3. Adaptive Integration
    print("3. ADAPTIVE INTEGRATION (Adaptive Simpson)")
    print("-" * 80)
    
    adaptive2 = adaptive_simpson(f2, a2, b2)
    print(f"Hasil: {adaptive2:.15f}")
    print(f"Error: {abs(adaptive2 - exact2):.6e}")
    print()
    
    # 4. Gaussian Quadrature
    print("4. GAUSSIAN QUADRATURE (5-point)")
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
            'Romberg R₄,₄',
            'Adaptive Simpson',
            'Gaussian Quadrature'
        ],
        'Hasil': [
            trap2_R00,
            trap2_R11,
            trap2_R22,
            romberg2[4][4],
            adaptive2,
            gauss2
        ],
        'Error': [
            abs(trap2_R00 - exact2),
            abs(trap2_R11 - exact2),
            abs(trap2_R22 - exact2),
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
    print()
    
    # Buat laporan PDF dengan langkah-langkah matematis
    print("="*80)
    print("MEMBUAT LAPORAN PDF...")
    print("="*80)
    try:
        pdf_file = create_pdf_report(results1, results2, romberg1, romberg2, exact1, exact2)
        print(f"✓ Laporan PDF berhasil dibuat: {pdf_file}")
        print("  Laporan ini berisi:")
        print("  - Penjelasan teori setiap metode")
        print("  - Langkah-langkah perhitungan matematis detail")
        print("  - Tabel hasil lengkap")
        print("  - Analisis dan perbandingan metode")
    except Exception as e:
        print(f"✗ Error membuat PDF: {e}")
    print("="*80)


if __name__ == "__main__":
    main()