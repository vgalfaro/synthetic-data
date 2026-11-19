"""Redondeo controlado de W* a enteros Y (ver 09_rounding.ipynb).

    y = floor(W*) + z,  z en {0,1} solo en las celdas "libres" (parte fraccional >= tau o entre las L mayores de su fila)
    duras:   sum_m y[p,m] = mu_p,   sum_p y[p,m] = nu_m
    blandas: sum_{p: x_pk = c} y[p,m] - d+ + d- = nu_k[c,m]  para cada bloque k y (c,m) observado
    objetivo: min sum (d+ + d-)

Versión dura (duro=True): los bloques son igualdades, sin d; el objetivo es 0 (factibilidad) o la cercanía a W*
(min sum (1 - 2 frac) z, que equivale a min sum |y - W*|). Para que no sea infactible de entrada, las celdas libres se
eligen con celdas_libres_duras (repara las restricciones de bloque que no alcanzan su valor del censo).

Se resuelve con HiGHS (highspy), o se escribe un .mps para resolverlo con otro solver (Gurobi) y leer su .sol.
Solución inicial (versión blanda): flujo máximo con los perfiles en orden aleatorio.
"""
import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import maximum_flow

from .perfiles import codigos_bloque, indicadora

EPS = 1e-6  # W* es float32: valores a menos de EPS de un entero se toman como enteros


def partes(W, mu, nu):
    # floor de W* y residuos por fila (r_p) y columna (c_m) que deben llenar los z
    W = np.asarray(W, dtype=np.float64)
    W = np.where(np.abs(W - np.round(W)) < EPS, np.round(W), W)
    F = np.floor(W)
    frac = W - F
    r = np.round(mu).astype(np.int64) - F.sum(axis=1).astype(np.int64)
    c = np.round(nu).astype(np.int64) - F.sum(axis=0).astype(np.int64)
    assert (r >= 0).all() and (c >= 0).all() and r.sum() == c.sum()
    return F, frac, r, c


def celdas_libres(frac, tau, L):
    # celdas con parte fraccional >= tau, más las L de mayor parte fraccional en cada fila (si son > 0)
    libre = frac >= tau
    if L > 0:
        top = np.argpartition(-frac, min(L, frac.shape[1] - 1), axis=1)[:, :L]
        filas = np.repeat(np.arange(frac.shape[0]), L)
        sel = np.zeros_like(libre)
        sel[filas, top.ravel()] = True
        libre |= sel & (frac > EPS)
    return libre & (frac > EPS)


def condicion_necesaria(F, libre, inv, modelo):
    # Para cada bloque: sum_{p: x_pk=c} floor <= nu_k[c,m] <= sum floor + #libres. Devuelve {bloque: (falta, sobra)},
    # con falta = max(0, nu_k - máximo alcanzable) y sobra = max(0, mínimo - nu_k), ambos (C, M) con 0 en celdas NaN
    out = {}
    for j, b in enumerate(modelo["bloques"]):
        nu_k = codigos_bloque(inv, b)[1]
        I = indicadora(modelo["perfiles"][:, j], nu_k.shape[0])
        lo = np.asarray(I @ F)
        hi = lo + np.asarray(I @ libre.astype(np.float64))
        nk = np.nan_to_num(nu_k, nan=0.0)
        obs = ~np.isnan(nu_k)
        out[b] = (np.where(obs, np.maximum(0, nk - hi), 0), np.where(obs, np.maximum(0, lo - nk), 0))
    return out


def celdas_libres_duras(F, frac, inv, modelo, tau, L):
    # celdas_libres(tau, L) más, en cada restricción de bloque que no alcanza su valor del censo, las celdas de mayor
    # parte fraccional de ese grupo (perfiles con esa categoría, en esa manzana) hasta que alcance
    libre = celdas_libres(frac, tau, L)
    for j, b in enumerate(modelo["bloques"]):
        falta = condicion_necesaria(F, libre, inv, {"bloques": [b], "perfiles": modelo["perfiles"][:, [j]]})[b][0]
        cat = modelo["perfiles"][:, j]
        for cc, m in zip(*np.nonzero(falta)):
            cand = np.nonzero((cat == cc) & ~libre[:, m] & (frac[:, m] > EPS))[0]
            elegidos = cand[np.argsort(-frac[cand, m])[:int(round(falta[cc, m]))]]
            libre[elegidos, m] = True
    return libre


def flujo(r, c, pi, mi, seed=None):
    # flujo máximo fuente -> perfil (r) -> manzana (arcos de capacidad 1 en (pi, mi)) -> sumidero (c).
    # Con seed, los perfiles se numeran en orden aleatorio para no favorecer un rango de índices. Devuelve (z, saturó)
    P, M = len(r), len(c)
    perm = np.arange(P) if seed is None else np.random.default_rng(seed).permutation(P)
    s, t = 0, P + M + 1
    fil = np.r_[np.zeros(P, np.int64), 1 + perm[pi], 1 + P + np.arange(M)]
    col = np.r_[1 + perm, 1 + P + mi, np.full(M, t)]
    cap = np.r_[r, np.ones(len(pi), np.int64), c].astype(np.int32)
    G = sp.csr_matrix((cap, (fil, col)), shape=(t + 1, t + 1))
    res = maximum_flow(G, s, t)
    z = np.asarray(res.flow[1 + perm[pi], 1 + P + mi]).ravel().astype(np.int64)
    return z, res.flow_value == r.sum()


def construir(W, mu, nu, inv, modelo, tau=0.05, L=3, libre=None, duro=False, objetivo="cero"):
    # Arma el MIP en forma dispersa. Variables: [z (n_z, binarias), d+ (B), d- (B)]; en la versión dura solo z.
    # libre: máscara (P, M) de celdas libres; si es None se usa celdas_libres(tau, L).
    # objetivo (solo duro): "cero" (factibilidad) o "cercania" (min sum (1 - 2 frac) z)
    F, frac, r, c = partes(W, mu, nu)
    if libre is None:
        libre = celdas_libres(frac, tau, L)
    pi, mi = np.nonzero(libre)
    P, M = F.shape
    n_z = len(pi)
    filas, cols, rhs = [], [], []
    # filas mu_p y columnas nu_m (duras)
    filas += [pi, P + mi]
    cols += [np.arange(n_z)] * 2
    rhs += [r, c]
    n_fil = P + M
    # bloques (blandas)
    bloques, b_rhs, b_info = modelo["bloques"], [], []
    for j, b in enumerate(bloques):
        nu_k = codigos_bloque(inv, b)[1]
        cat = modelo["perfiles"][:, j]
        obs = ~np.isnan(nu_k)
        ids = np.full(nu_k.shape, -1, dtype=np.int64)
        ids[obs] = n_fil + np.arange(obs.sum())
        proyF = np.asarray(indicadora(cat, nu_k.shape[0]) @ F)
        b_rhs.append((nu_k - proyF)[obs])
        b_info += [(b, cc, m) for cc, m in zip(*np.nonzero(obs))]
        fila_j = ids[cat[pi], mi]
        ok = fila_j >= 0
        filas.append(fila_j[ok])
        cols.append(np.nonzero(ok)[0])
        n_fil += obs.sum()
    B = n_fil - P - M
    b_rhs = np.concatenate(b_rhs)
    n_d = 0 if duro else 2 * B
    if not duro:
        filas.append(P + M + np.arange(B)); cols.append(n_z + np.arange(B))       # - d+
        filas.append(P + M + np.arange(B)); cols.append(n_z + B + np.arange(B))   # + d-
    n_uno = sum(len(x) for x in (filas if duro else filas[:-2]))
    vals = np.r_[np.ones(n_uno), -np.ones(n_d // 2), np.ones(n_d // 2)]
    A = sp.csc_matrix((vals, (np.concatenate(filas), np.concatenate(cols))), shape=(n_fil, n_z + n_d))
    rhs = np.r_[r, c, b_rhs].astype(np.float64)
    if duro:
        costo = 1 - 2 * frac[pi, mi] if objetivo == "cercania" else np.zeros(n_z)
    else:
        costo = np.r_[np.zeros(n_z), np.ones(n_d)]
    return {"A": A, "rhs": rhs, "costo": costo, "n_z": n_z, "B": B, "P": P, "M": M, "pi": pi, "mi": mi, "F": F,
            "r": r, "c": c, "b_info": b_info, "tau": tau, "L": L, "duro": duro, "objetivo": objetivo,
            "frac_libre": frac[pi, mi]}


def objetivo_de(mip, z):
    # desviación total sum |d| de una z, y el vector de desviaciones de bloque (proyección - nu_k)
    A, n_z, P, M = mip["A"], mip["n_z"], mip["P"], mip["M"]
    dev = A[P + M:, :n_z] @ z - mip["rhs"][P + M:]
    return float(np.abs(dev).sum()), dev


def solucion_inicial(mip, seeds=range(5)):
    # mejor de varios flujos máximos con perfiles en orden aleatorio
    mejor = None
    for s in seeds:
        z, ok = flujo(mip["r"], mip["c"], mip["pi"], mip["mi"], seed=s)
        assert ok, "el flujo no satura: las celdas libres no alcanzan para mu/nu (bajar tau o subir L)"
        obj = objetivo_de(mip, z)[0]
        if mejor is None or obj < mejor[0]:
            mejor = (obj, z)
    return mejor[1], mejor[0]


def modelo_highs(mip, nombres=False):
    # Objeto Highs con el MIP cargado (z binarias; d continuas >= 0 en la versión blanda)
    import highspy
    A, rhs, n_z = mip["A"], mip["rhs"], mip["n_z"]
    n = A.shape[1]
    h = highspy.Highs()
    lp = highspy.HighsLp()
    lp.num_col_, lp.num_row_ = n, A.shape[0]
    lp.col_cost_ = np.asarray(mip["costo"], dtype=np.float64)
    lp.col_lower_ = np.zeros(n)
    lp.col_upper_ = np.r_[np.ones(n_z), np.full(n - n_z, highspy.kHighsInf)]
    lp.row_lower_ = rhs
    lp.row_upper_ = rhs
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_, lp.a_matrix_.index_, lp.a_matrix_.value_ = A.indptr, A.indices, A.data
    lp.integrality_ = [highspy.HighsVarType.kInteger] * n_z + [highspy.HighsVarType.kContinuous] * (n - n_z)
    if nombres:
        lp.col_names_ = [f"z{j}" for j in range(n_z)] + [f"d{j}" for j in range(n - n_z)]
        lp.row_names_ = [f"r{i}" for i in range(A.shape[0])]
    h.passModel(lp)
    return h


def escribir_mps(mip, ruta):
    # Escribe el MIP en formato MPS (variables z0, z1, ... en el orden de mip["pi"], mip["mi"]) para otro solver
    h = modelo_highs(mip, nombres=True)
    h.setOptionValue("output_flag", False)
    return h.writeModel(ruta)


def leer_sol(mip, ruta):
    # z desde un archivo .sol (formato Gurobi: "nombre valor" por línea, comentarios con #)
    z = np.zeros(mip["n_z"], dtype=np.int64)
    with open(ruta) as f:
        for linea in f:
            if linea.startswith("z"):
                nombre, valor = linea.split()[:2]
                z[int(nombre[1:])] = int(round(float(valor)))
    return z


def resolver(mip, z0=None, horas=6.0, log=None, gap=0.0, guardar=None, cada=300):
    # Resuelve con HiGHS. guardar(z, obj) se llama con cada mejor solución (como máximo una vez cada `cada` segundos,
    # salvo la última) para no perder lo avanzado si se corta
    import highspy
    n_z = mip["n_z"]
    h = modelo_highs(mip)
    h.setOptionValue("time_limit", float(horas * 3600))
    h.setOptionValue("mip_rel_gap", float(gap))
    if log:
        h.setOptionValue("log_file", log)
        h.setOptionValue("log_to_console", False)
    if z0 is not None and not mip["duro"]:
        dev = objetivo_de(mip, z0)[1]
        sol = highspy.HighsSolution()
        sol.col_value = np.r_[z0, np.maximum(dev, 0), np.maximum(-dev, 0)].astype(np.float64)
        sol.value_valid = True
        h.setSolution(sol)
    ultimo = [0.0]
    if guardar is not None:
        def cb(e):
            if time.time() - ultimo[0] >= cada:
                ultimo[0] = time.time()
                guardar(np.round(np.asarray(e.data_out.mip_solution)[:n_z]).astype(np.int64),
                        e.data_out.objective_function_value)
        h.cbMipImprovingSolution.subscribe(cb)
    t0 = time.time()
    h.run()
    info = h.getInfo()
    z = np.round(np.asarray(h.getSolution().col_value)[:n_z]).astype(np.int64)
    tiene_sol = info.primal_solution_status == 2  # 2 = solución factible
    res = {"estado": str(h.getModelStatus()), "tiene_solucion": bool(tiene_sol),
           "objetivo": info.objective_function_value, "cota": info.mip_dual_bound, "gap": info.mip_gap,
           "segundos": time.time() - t0}
    if not tiene_sol:
        return None, res
    if guardar is not None:
        guardar(z, res["objetivo"])
    return z, res


def resolver_gurobi(mip, z0=None, horas=24.0, log=None, gap=0.0, guardar=None, cada=300, hilos=0, metodo=2,
                    ruta_iis=None):
    # Mismo MIP con Gurobi. metodo: algoritmo del LP raíz (2 = barrera, paralela; -1 = automático).
    # En la versión dura se usa MIPFocus=1 (buscar factibilidad). Si es infactible y hay ruta_iis, calcula un IIS
    # (subconjunto irreducible de restricciones en conflicto) y lo escribe en formato .ilp
    import gurobipy as gp
    from gurobipy import GRB
    A, rhs, n_z = mip["A"], mip["rhs"], mip["n_z"]
    n = A.shape[1]
    m = _modelo_gurobi(log=log, salida=False)
    x = m.addMVar(n, lb=0.0, ub=np.r_[np.ones(n_z), np.full(n - n_z, GRB.INFINITY)],
                  vtype=np.array([GRB.BINARY] * n_z + [GRB.CONTINUOUS] * (n - n_z)))
    m.addMConstr(A.tocsr(), x, "=", rhs)
    m.setObjective(np.asarray(mip["costo"], dtype=np.float64) @ x, GRB.MINIMIZE)
    m.Params.TimeLimit = float(horas * 3600)
    m.Params.MIPGap = float(gap)
    m.Params.Threads = int(hilos)
    m.Params.Method = int(metodo)
    if mip["duro"]:
        m.Params.MIPFocus = 1
    if z0 is not None:
        x.Start = np.r_[z0, np.full(n - n_z, GRB.UNDEFINED)].astype(np.float64)
    ultimo = [0.0]

    def cb(model, where):
        if guardar is not None and where == GRB.Callback.MIPSOL and time.time() - ultimo[0] >= cada:
            ultimo[0] = time.time()
            z = np.round(model.cbGetSolution(x)[:n_z]).astype(np.int64)
            guardar(z, model.cbGet(GRB.Callback.MIPSOL_OBJ))

    t0 = time.time()
    m.optimize(cb)
    estados = {GRB.OPTIMAL: "optimo", GRB.INFEASIBLE: "infactible", GRB.TIME_LIMIT: "limite_tiempo",
               GRB.INF_OR_UNBD: "infactible_o_no_acotado", GRB.INTERRUPTED: "interrumpido"}
    res = {"estado": estados.get(m.Status, str(m.Status)), "tiene_solucion": m.SolCount > 0,
           "objetivo": m.ObjVal if m.SolCount > 0 else float("inf"),
           "cota": m.ObjBound if m.Status != GRB.INFEASIBLE else float("inf"),
           "gap": m.MIPGap if m.SolCount > 0 else float("inf"), "nodos": m.NodeCount, "segundos": time.time() - t0}
    if m.Status == GRB.INFEASIBLE and ruta_iis:
        m.Params.TimeLimit = float(horas * 3600)
        m.computeIIS()
        m.write(ruta_iis)
        res["iis_filas"] = int(sum(c.IISConstr for c in m.getConstrs()))
        res["iis"] = ruta_iis
    if m.SolCount == 0:
        return None, res
    z = np.round(x.X[:n_z]).astype(np.int64)
    if guardar is not None:
        guardar(z, res["objetivo"])
    return z, res


def verificar(mip, Y, mu, nu, inv, modelo):
    # Chequeo exacto de una Y entera: filas, columnas, cotas floor/ceil y cada restricción de bloque observada
    W_floor = mip["F"]
    res = {"filas = mu_p": bool((Y.sum(axis=1) == mu).all()), "columnas = nu_m": bool((Y.sum(axis=0) == nu).all()),
           "floor <= Y <= floor + 1": bool(((Y >= W_floor) & (Y <= W_floor + 1)).all())}
    viol, peor = 0, 0.0
    for j, b in enumerate(modelo["bloques"]):
        nu_k = codigos_bloque(inv, b)[1]
        dif = np.asarray(indicadora(modelo["perfiles"][:, j], nu_k.shape[0]) @ Y) - nu_k
        dif = np.abs(dif[~np.isnan(nu_k)])
        viol += int((dif > 0.5).sum())
        peor = max(peor, float(dif.max()))
    res["restricciones de bloque violadas"] = viol
    res["peor |proyección - nu_k|"] = peor
    return res


def errores_bloque(inv, modelo, Y):
    # Por bloque, sobre las celdas (c, m) observadas: error absoluto máximo, medio y % de celdas con |error| <= 1
    out = {}
    Y = np.asarray(Y, dtype=np.float64)
    for j, b in enumerate(modelo["bloques"]):
        nu_k = codigos_bloque(inv, b)[1]
        dif = np.abs(np.asarray(indicadora(modelo["perfiles"][:, j], nu_k.shape[0]) @ Y) - nu_k)[~np.isnan(nu_k)]
        out[b] = {"max": float(dif.max()), "medio": float(dif.mean()), "pct_<=1": float(100 * (dif <= 1).mean())}
    return out


def nombre(tipo, unidad, cut, duro, tau, L):
    # nombre de archivo de una corrida: <tipo>_<unidad>_<cut>_<duro|blando>_t<tau>_L<L> (sin extensión)
    return f"{tipo}_{unidad}_{cut}_{'duro' if duro else 'blando'}_t{tau:g}_L{L}"


def matriz_Y(mip, z):
    # Y entera (P, M) = floor(W*) + z
    Y = mip["F"].astype(np.int64)
    Y[mip["pi"], mip["mi"]] += z
    return Y


# ---------------------------------------------------------------------------------------------------------------------
# Muestreo de un punto extremo con E[Y] = W* (dependent rounding en el grafo bipartito perfiles-manzanas;
# Gandhi, Khuller, Parthasarathy y Srinivasan 2006). Es el procedimiento de la pizarra: el PL de transporte
# (filas mu_p, columnas nu_m, cotas floor/ceil) tiene puntos extremos enteros y W* es combinación convexa de ellos;
# este algoritmo sortea uno con la distribución correcta sin calcular la descomposición. Garantías:
#   (1) floor(W*) <= Y <= ceil(W*)   (2) E[Y] = W*   (3) filas = mu_p y columnas = nu_m exactas
# y además correlación negativa entre celdas de una misma manzana, así que cada marginal de bloque (suma de celdas de
# una manzana) se concentra alrededor de su esperanza.
# ---------------------------------------------------------------------------------------------------------------------

def ajustar_margenes(W, mu, nu, tol=1e-11, n_iter=10000):
    # IPF solo en filas y columnas (float64) para que W* cumpla mu_p y nu_m exactamente (requisito de (3) y (2))
    W = np.asarray(W, dtype=np.float64).copy()
    mu = np.asarray(mu, dtype=np.float64)
    nu = np.asarray(nu, dtype=np.float64)
    for it in range(n_iter):
        f = W.sum(axis=1)
        W *= np.divide(mu, f, out=np.zeros_like(mu), where=f > 0)[:, None]
        g = W.sum(axis=0)
        W *= np.divide(nu, g, out=np.zeros_like(nu), where=g > 0)[None, :]
        if np.abs(W.sum(axis=1) - mu).max() < tol:
            break
    return W, it + 1


def _numba():
    from numba import njit

    @njit(cache=True)
    def fraccional(x, e, eps):
        return eps < x[e] < 1.0 - eps

    @njit(cache=True)
    def siguiente(u, excluir, x, ptr, adj, pos, eps):
        # primera arista fraccional de u distinta de `excluir`; avanza pos[u] sobre las que ya son enteras
        k = pos[u]
        while k < ptr[u + 1] and not fraccional(x, adj[k], eps):
            k += 1
        pos[u] = k
        while k < ptr[u + 1]:
            e = adj[k]
            if e != excluir and fraccional(x, e, eps):
                return e
            k += 1
        return -1

    @njit(cache=True)
    def redondeo_dependiente(eu, ev, x, ptr, adj, n_nodos, seed, eps):
        # x: partes fraccionales (se modifica). eu/ev: extremos de cada arista (ids de nodo). ptr/adj: listas de
        # adyacencia (CSR). Recorre caminos de aristas fraccionales; al cerrar un ciclo (par, el grafo es bipartito)
        # suma +a/-a alternado, con la dirección al azar de modo que E[x] no cambie; al menos una arista queda entera.
        np.random.seed(seed)
        pos = ptr[:-1].copy()
        en_camino = np.full(n_nodos, -1, dtype=np.int64)
        camino_n = np.empty(n_nodos + 1, dtype=np.int64)
        camino_e = np.empty(n_nodos + 1, dtype=np.int64)
        ciclos = 0
        for s in range(n_nodos):
            L = 0
            camino_n[0] = s
            camino_e[0] = -1
            en_camino[s] = 0
            while L >= 0:
                u = camino_n[L]
                e = siguiente(u, camino_e[L], x, ptr, adj, pos, eps)
                if e == -1:
                    # sin salida: con márgenes enteros solo pasa por error de redondeo; la arista de llegada se
                    # redondea al entero más cercano (cambio del orden de eps)
                    ein = camino_e[L]
                    if ein >= 0 and fraccional(x, ein, eps):
                        x[ein] = np.round(x[ein])
                    en_camino[u] = -1
                    L -= 1
                    continue
                w = ev[e] if eu[e] == u else eu[e]
                if en_camino[w] >= 0:
                    i = en_camino[w]
                    # ciclo: aristas camino_e[i+1..L] y e, con signos +, -, +, ...
                    a = 1e300
                    b = 1e300
                    signo = 1
                    for j in range(i + 1, L + 2):
                        ej = camino_e[j] if j <= L else e
                        if signo > 0:
                            a = min(a, 1.0 - x[ej])
                            b = min(b, x[ej])
                        else:
                            a = min(a, x[ej])
                            b = min(b, 1.0 - x[ej])
                        signo = -signo
                    d = a if np.random.random() < b / (a + b) else -b
                    signo = 1
                    for j in range(i + 1, L + 2):
                        ej = camino_e[j] if j <= L else e
                        x[ej] += signo * d
                        if x[ej] < eps:
                            x[ej] = 0.0
                        elif x[ej] > 1.0 - eps:
                            x[ej] = 1.0
                        signo = -signo
                    ciclos += 1
                    for j in range(i + 1, L + 1):
                        en_camino[camino_n[j]] = -1
                    L = i
                else:
                    L += 1
                    camino_n[L] = w
                    camino_e[L] = e
                    en_camino[w] = L
        return ciclos

    return redondeo_dependiente


_REDONDEO_DEPENDIENTE = None


def muestrear(W, mu, nu, seed=0, eps=1e-12, W_ajustada=None):
    # Y entera con E[Y] = W* (ver arriba). W_ajustada: resultado de ajustar_margenes, para no repetirlo en cada muestra.
    # Devuelve (Y, info)
    global _REDONDEO_DEPENDIENTE
    if _REDONDEO_DEPENDIENTE is None:
        _REDONDEO_DEPENDIENTE = _numba()
    t0 = time.time()
    Wa = ajustar_margenes(W, mu, nu)[0] if W_ajustada is None else W_ajustada
    F = np.floor(Wa)
    frac = Wa - F
    P, M = Wa.shape
    pi, mi = np.nonzero(frac > 0)
    x = frac[pi, mi].copy()
    eu, ev = pi.astype(np.int64), (P + mi).astype(np.int64)
    n = P + M
    nodos = np.r_[eu, ev]
    orden = np.argsort(nodos, kind="stable")
    adj = (orden % len(x)).astype(np.int64)
    ptr = np.r_[0, np.cumsum(np.bincount(nodos, minlength=n))].astype(np.int64)
    ciclos = _REDONDEO_DEPENDIENTE(eu, ev, x, ptr, adj, n, int(seed), eps)
    Y = F.astype(np.int64)
    Y[pi, mi] += np.round(x).astype(np.int64)
    return Y, {"aristas": len(x), "ciclos": int(ciclos), "segundos": time.time() - t0,
               "fraccionales_al_final": int(((x > eps) & (x < 1 - eps)).sum())}


# ---------------------------------------------------------------------------------------------------------------------
# Redondeo iterativo de Cembrano, Correa y Verdugo (PNAS 2022, Teorema 5 y Algoritmos 1-2), estilo Beck-Fiala.
# Hipergrafo d-partito: vértices = restricciones (filas de mip["A"]), hiperaristas = celdas libres (columnas de A).
# Partes: perfiles (filas mu_p), manzanas (columnas nu_m) y cada uno de los bloques. Cada celda toca a lo más un
# vértice por parte. Dado x en [0,1]^E con A x entero (un punto exactamente factible de la relajación de la versión
# dura) y enteros u_l >= 0 con sum_l 1/(u_l + 2) <= 1, se obtiene z en {0,1}^E con |(A (z - x))_v| <= u_l para todo
# vértice v de la parte l, y z_e = x_e donde x_e ya es entero.
# ---------------------------------------------------------------------------------------------------------------------

def partes_filas(mip):
    # Parte de cada fila de A: "perfiles", "manzanas" o el nombre del bloque
    P, M = mip["P"], mip["M"]
    return np.array(["perfiles"] * P + ["manzanas"] * M + [b for b, _, _ in mip["b_info"]], dtype=object)


def presupuesto(u_partes):
    # sum_l 1 / (u_l + 2): debe ser <= 1 para la garantía del Teorema 5
    return float(sum(1.0 / (u + 2) for u in u_partes.values()))


_ENV_GUROBI = None


def entorno_gurobi():
    # Un solo entorno de Gurobi por proceso: la licencia WLS pide un token al crear el entorno, y desde este servidor
    # eso pasa por el túnel del Mac (el DNS está roto). Reusar el entorno evita pedir un token por cada LP; el token
    # dura WLSTokenDuration minutos y Gurobi lo renueva solo
    global _ENV_GUROBI
    if _ENV_GUROBI is None:
        import gurobipy as gp
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.setParam("WLSTokenDuration", 60)
        env.start()
        _ENV_GUROBI = env
    return _ENV_GUROBI


def _modelo_gurobi(log=None, salida=True):
    # Modelo vacío en el entorno compartido, con su propio log
    import gurobipy as gp
    m = gp.Model(env=entorno_gurobi())
    m.Params.OutputFlag = 1 if (salida or log) else 0
    if log:
        m.Params.LogFile = log
        m.Params.LogToConsole = 0
    return m


def _modelo_lp_gurobi(A, rhs, costo, log=None, hilos=0, metodo=-1, salida=True):
    from gurobipy import GRB
    env = None
    m = _modelo_gurobi(log=log, salida=salida)
    x = m.addMVar(A.shape[1], lb=0.0, ub=1.0)
    m.addMConstr(A.tocsr(), x, "=", rhs)
    m.setObjective(np.asarray(costo, dtype=np.float64) @ x, GRB.MINIMIZE)
    m.Params.Threads = int(hilos)
    m.Params.Method = int(metodo)
    return m, x, env


def vertice_lp(mip, log=None, hilos=0, horas=24.0):
    # Punto extremo de la relajación lineal de la versión dura (z en [0,1]), con barrera + crossover (Algoritmo 2,
    # paso 1). Devuelve (x, info); x = None si la relajación es infactible
    from gurobipy import GRB
    A, rhs, n_z = mip["A"], mip["rhs"], mip["n_z"]
    assert A.shape[1] == n_z, "usar la versión dura (sin variables d)"
    m, x, env = _modelo_lp_gurobi(A, rhs, mip["costo"], log=log, hilos=hilos, metodo=2)
    m.Params.TimeLimit = float(horas * 3600)
    m.Params.Crossover = -1  # crossover activado: la solución es básica (punto extremo)
    t0 = time.time()
    m.optimize()
    info = {"estado": m.Status, "segundos": time.time() - t0}
    if m.Status != GRB.OPTIMAL:
        return None, info
    xv = np.clip(np.asarray(x.X, dtype=np.float64), 0.0, 1.0)
    info.update({"objetivo": m.ObjVal, "fraccionales": int(((xv > 1e-9) & (xv < 1 - 1e-9)).sum()),
                 "max_residuo": float(np.abs(A @ xv - rhs).max())})
    return xv, info


def redondeo_iterativo(mip, x, u_partes, tol=1e-6, hilos=0, log=None, reportar=print, estricto=True):
    # tol = tolerancia de factibilidad de Gurobi (1e-6): con 1e-9, una arista que el LP deja en 1 - 1.3e-9 seguía
    # "fraccional", su perfil (con una sola arista fraccional) salía de Q y el LP siguiente la llevaba a 0
    # Algoritmo 1 de Cembrano et al. sobre las hiperaristas fraccionales de x.
    #   u_partes: {"perfiles": u, "manzanas": u, <bloque>: u, ...}; una parte ausente no se controla (u = infinito).
    # En cada iteración: Q = vértices de cada parte l con >= u_l + 2 aristas fraccionales; se toma un punto extremo
    # del politopo {0 <= y <= 1 en las aristas fraccionales, (A y)_v = (A y_actual)_v para v en Q} (con el costo de
    # mip["costo"], cercanía a W* si objetivo="cercania"); las aristas que quedan enteras se fijan. Al final, las
    # fraccionales restantes se redondean al entero más cercano. Devuelve (z, historial)
    from gurobipy import GRB
    A = mip["A"].tocsr()
    n_z = mip["n_z"]
    parte = partes_filas(mip)
    umbral = np.array([u_partes[p] + 2 if p in u_partes else np.iinfo(np.int64).max for p in parte], dtype=np.int64)
    y = np.asarray(x, dtype=np.float64).copy()
    y[y < tol] = 0.0
    y[y > 1 - tol] = 1.0
    costo = np.asarray(mip["costo"], dtype=np.float64)
    historial = []
    t = 0
    while True:
        frac = np.nonzero((y > tol) & (y < 1 - tol))[0]
        if len(frac) == 0:
            break
        A_E = A[:, frac]
        grado = np.diff(A_E.tocsr().indptr)                  # aristas fraccionales por vértice
        Q = np.nonzero(grado >= umbral)[0]
        por_parte = {}
        for p in np.unique(parte[Q]):
            por_parte[p] = int((parte[Q] == p).sum())
        if len(Q) == 0:
            historial.append({"iter": t, "fraccionales": len(frac), "Q": 0})
            break
        A_QE = A_E[Q]
        rhs = A_QE @ y[frac]
        t0 = time.time()
        m, yv, env = _modelo_lp_gurobi(A_QE, rhs, costo[frac], log=log, hilos=hilos, metodo=-1, salida=log is not None)
        m.optimize()
        if m.Status != GRB.OPTIMAL:
            raise RuntimeError(f"LP de la iteración {t} terminó con estado {m.Status}")
        nuevo = np.clip(np.asarray(yv.X, dtype=np.float64), 0.0, 1.0)
        nuevo[nuevo < tol] = 0.0
        nuevo[nuevo > 1 - tol] = 1.0
        y[frac] = nuevo
        fijadas = int(((nuevo == 0) | (nuevo == 1)).sum())
        h = {"iter": t, "fraccionales": len(frac), "Q": len(Q), "fijadas": fijadas, "segundos": time.time() - t0,
             "Q_por_parte": por_parte}
        historial.append(h)
        if reportar:
            reportar(f"iter {t}: {len(frac):,} fraccionales, |Q| = {len(Q):,}, fijadas {fijadas:,} "
                     f"({h['segundos']:.1f} s)")
        m.dispose()
        if fijadas == 0:
            if estricto:
                raise RuntimeError(f"la iteración {t} no fijó ninguna arista (revisar u_partes: presupuesto "
                                   f"{presupuesto(u_partes):.3f})")
            h["sin_progreso"] = True   # con x de sumas no enteras (versión blanda) puede pasar: se redondea lo que queda
            break
        t += 1
    z = np.round(y).astype(np.int64)
    return z, historial


def desviaciones_por_parte(mip, z, x):
    # max |(A (z - x))_v| por parte: lo que el Teorema 5 acota por u_l
    dev = np.abs(mip["A"].tocsr() @ (z - x))
    parte = partes_filas(mip)
    return {p: float(dev[parte == p].max()) for p in dict.fromkeys(parte)}


# ---------------------------------------------------------------------------------------------------------------------
# Comunas grandes: división en grupos de manzanas (el LP completo no cabe en memoria).
#   1. Grupos de manzanas contiguas (orden de las columnas de W) con a lo más max_celdas celdas no nulas cada uno.
#   2. Reparto de mu_p entre grupos: a[p, g] = sum_{m en g} W*_pm tiene filas mu_p y columnas T_g = sum_{m en g} nu_m
#      (enteros), así que se redondea con el muestreo de transporte (filas y columnas exactas, E = a).
#   3. Cada grupo es un subproblema con filas mu_pg, columnas nu_m y los bloques de sus manzanas (que no cruzan grupos).
#      Como el reparto quedó fijo, el LP exacto del grupo podría no tener solución: se usa la versión blanda.
# ---------------------------------------------------------------------------------------------------------------------

def particion_manzanas(W, max_celdas):
    # Lista de arreglos de índices de columnas (manzanas contiguas) con <= max_celdas celdas no nulas por grupo
    nnz = np.asarray((np.asarray(W) > 0).sum(axis=0), dtype=np.int64)
    grupos, actual, suma = [], [], 0
    for m, k in enumerate(nnz):
        if actual and suma + k > max_celdas:
            grupos.append(np.array(actual))
            actual, suma = [], 0
        actual.append(m)
        suma += k
    if actual:
        grupos.append(np.array(actual))
    return grupos


def repartir_perfiles(Wa, mu, nu, grupos, seed=0):
    # mu_pg enteros (P, G) con filas mu_p y columnas sum_{m en g} nu_m exactas, E[mu_pg] = sum_{m en g} Wa_pm
    a = np.column_stack([Wa[:, g].sum(axis=1) for g in grupos])
    T = np.array([int(round(np.asarray(nu)[g].sum())) for g in grupos])
    mu_g, _ = muestrear(a, mu, T, seed=seed)
    assert (mu_g.sum(axis=1) == np.asarray(mu)).all() and (mu_g.sum(axis=0) == T).all()
    return mu_g


def subinventario(inv, cols):
    # Inventario restringido a las manzanas `cols` (solo lo que usa codigos_bloque)
    from types import SimpleNamespace
    marg = {b: {**info, "nu": np.asarray(info["nu"])[:, cols]} for b, info in inv.marginales.items()}
    return SimpleNamespace(marginales=marg, per=inv.per, nu=np.asarray(inv.nu)[cols])


def vertice_lp_blando(mip, peso_cercania=1e-3, log=None, hilos=0, horas=24.0):
    # Punto extremo del LP blando: min sum |d| + peso_cercania * sum (1 - 2 frac) z, z en [0,1], d >= 0.
    # Devuelve (z, d_total, info)
    import gurobipy as gp
    from gurobipy import GRB
    A, rhs, n_z = mip["A"], mip["rhs"], mip["n_z"]
    n = A.shape[1]
    F_frac = np.asarray(mip["frac_libre"], dtype=np.float64)
    costo = np.r_[peso_cercania * (1 - 2 * F_frac), np.ones(n - n_z)]
    m = _modelo_gurobi(log=log, salida=False)
    x = m.addMVar(n, lb=0.0, ub=np.r_[np.ones(n_z), np.full(n - n_z, GRB.INFINITY)])
    m.addMConstr(A.tocsr(), x, "=", rhs)
    m.setObjective(costo @ x, GRB.MINIMIZE)
    m.Params.Threads = int(hilos)
    m.Params.Method = 2
    m.Params.Crossover = -1
    m.Params.TimeLimit = float(horas * 3600)
    t0 = time.time()
    m.optimize()
    if m.Status != GRB.OPTIMAL:
        return None, None, {"estado": m.Status, "segundos": time.time() - t0}
    xv = np.asarray(x.X, dtype=np.float64)
    z = np.clip(xv[:n_z], 0.0, 1.0)
    d_total = float(xv[n_z:].sum())
    return z, d_total, {"estado": m.Status, "segundos": time.time() - t0, "desviacion_lp": d_total,
                        "fraccionales": int(((z > 1e-9) & (z < 1 - 1e-9)).sum())}
