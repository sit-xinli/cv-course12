#the linux environment is required.
#uv pip install git+https://github.com/scomup/MathematicalRobotics.git#egg=mathR


#　カメラの頂点
class CameraVertex(BaseVertex):
    def __init__(self, x):
        super().__init__(x, 6)

    def update(self, dx):
        # x in SE(3)のため、指数展開より更新
        self.x = self.x @ expSE3(dx)

#　3D点の頂点
class PointVertex(BaseVertex):
    def __init__(self, x):
        super().__init__(x, 3)

    def update(self, dx):
        # x in R3 のため、ユークリッド空間における更新
        self.x = self.x + dx

# 再投影誤差のエッジ
class ReprojEdge(BaseEdge):
    def __init__(self, link, z, omega=np.eye(2), kernel=None):
        super().__init__(link, z, omega, kernel)

    def residual(self, vertices):
        Twc = vertices[self.link[0]].x
        pw = vertices[self.link[1]].x
        u, K = self.z
        pc, dpcdTwc, dpcdpw = transform_inv(Twc, pw, True)
        u_reproj, dudpc = reproject(pc, K, True)
        JTwc = dudpc @ dpcdTwc  # 式(6)
        Jpw = dudpc @ dpcdpw  # 式(7)
        return u_reproj-u, [JTwc, Jpw]  # 式(2)再投影誤差

# 関数T、世界座標系の3d点をカメラの座標系の3d点に変換
def transform_inv(x, p, calcJ=False):
    if x.shape[0] == 6:
        T = expSE3(x)
    else:
        T = x
    Tinv = np.linalg.inv(T)
    Rinv, tinv = makeRt(Tinv)
    r = Rinv @ p + tinv  # 式(3)
    if (calcJ is True):  #ヤコビ行列計算
        M1 = -np.eye(3)  #式(9)
        M2 = skew(r)  #式(9)
        dTdx = np.hstack([M1, M2])
        dTdp = Rinv  #式(10)
        return r, dTdx, dTdp
    else:
        return r

# 関数P、カメラの座標系の3d点を、画像の2d点に変換
def reproject(pc, K, calcJ=False):
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    x, y, z = pc
    z_2 = z * z
    r = np.array([(x * fx / z + cx),  # 式(4)
                  (y * fy / z + cy)])
    if (calcJ is True):  #ヤコビ行列計算　#式(8)
        J = np.array([[fx / z,    0, -fx * x / z_2],
                      [0, fy / z, -fy * y / z_2]])
        return r, J
    else:
        return r
