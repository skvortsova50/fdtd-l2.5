import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class FDTD2D_TMz:
    """
    2D FDTD для TMz хвиль (Ez, Hx, Hy)
    Без провідності, зі струмовим джерелом Jz
    """

    def __init__(self, Nx=120, Ny=120, dx=1e-3, dy=1e-3, Nt=600):
        # --- Параметри сітки ---
        self.Nx, self.Ny = Nx, Ny
        self.dx, self.dy = dx, dy
        S = 1
        self.dt = S / (np.sqrt((1/dx**2) + (1/dy**2)))
        self.Nt = Nt

        # Courant numbers
        self.Sx = self.dt / self.dx
        self.Sy = self.dt / self.dy

        # --- Поля (Yee-схема) ---
        self.Ez = np.zeros((Nx, Ny))
        self.Hx = np.zeros((Nx, Ny-1))
        self.Hy = np.zeros((Nx-1, Ny))

        self.Ez_prev = np.zeros_like(self.Ez)

        # --- Джерело ---
        self.src_x, self.src_y = Nx//2, Ny//2
        self.t0, self.spread = 40, 12

        # --- Датчик ---
        self.probe_x, self.probe_y = Nx//2 + 20, Ny//2
        self.signal = []

    # -------------------------------------------------
    def Jz(self, n):
        """Гаусовий імпульс струму"""
        return np.exp(-((n - self.t0) / self.spread)**2)

    # -------------------------------------------------
    def update_H(self):
        """Оновлення Hx, Hy"""
        self.Hx -= self.Sy * (self.Ez[:, 1:] - self.Ez[:, :-1])
        self.Hy += self.Sx * (self.Ez[1:, :] - self.Ez[:-1, :])

    # -------------------------------------------------
    def update_E(self):
        """Оновлення Ez (внутрішні вузли)"""
        curl_H = (
            (self.Hy[1:, 1:-1] - self.Hy[:-1, 1:-1]) / self.dx
            - (self.Hx[1:-1, 1:] - self.Hx[1:-1, :-1]) / self.dy
        )
        self.Ez[1:-1, 1:-1] += self.dt * curl_H

    # -------------------------------------------------
    def apply_mur_abc(self):
        """Mur ABC (1-й порядок)"""
        cx = (self.Sx - 1) / (self.Sx + 1)
        cy = (self.Sy - 1) / (self.Sy + 1)

        # left
        self.Ez[0, 1:-1] = (
            self.Ez_prev[1, 1:-1]
            + cx * (self.Ez[1, 1:-1] - self.Ez_prev[0, 1:-1])
        )
        # right
        self.Ez[-1, 1:-1] = (
            self.Ez_prev[-2, 1:-1]
            + cx * (self.Ez[-2, 1:-1] - self.Ez_prev[-1, 1:-1])
        )

        # bottom
        self.Ez[1:-1, 0] = (
            self.Ez_prev[1:-1, 1]
            + cy * (self.Ez[1:-1, 1] - self.Ez_prev[1:-1, 0])
        )
        # top
        self.Ez[1:-1, -1] = (
            self.Ez_prev[1:-1, -2]
            + cy * (self.Ez[1:-1, -2] - self.Ez_prev[1:-1, -1])
        )

        # кути
        self.Ez[0, 0] = self.Ez[1, 1]
        self.Ez[0, -1] = self.Ez[1, -2]
        self.Ez[-1, 0] = self.Ez[-2, 1]
        self.Ez[-1, -1] = self.Ez[-2, -2]

    # -------------------------------------------------
    def source(self, n):
        """Струмове джерело Jz"""
        self.Ez[self.src_x, self.src_y] -= self.dt * self.Jz(n)

    # -------------------------------------------------
    def step(self, n):
        """Один часовий крок"""
        self.Ez_prev[:, :] = self.Ez
        self.update_H()
        self.update_E()
        self.apply_mur_abc()
        self.source(n)
        self.signal.append(self.Ez[self.probe_x, self.probe_y])

    # -------------------------------------------------
    def animate(self):
        """Анімація поширення поля"""

        fig, ax = plt.subplots()
        im = ax.imshow(self.Ez.T, cmap="RdBu_r",
                       vmin=-1, vmax=1, origin="lower")
        plt.colorbar(im, ax=ax)
        ax.set_title("2D TMz FDTD: Ez(x,y)")

        def update(n):
            self.step(n)
            im.set_array(15000 * self.Ez.T)
            ax.set_title(f"2D TMz FDTD: Ez(x,y), n = {n}")
            return [im]

        ani = animation.FuncAnimation(
            fig, update, frames=self.Nt, interval=20
        )
        plt.show()


# =====================================================
# ЗАПУСК ЛАБОРАТОРНОЇ
# =====================================================

if __name__ == "__main__":
    sim = FDTD2D_TMz(Nx=120, Ny=120, Nt=200)
    sim.animate()
