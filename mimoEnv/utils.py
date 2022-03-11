import numpy as np
from matplotlib import pyplot as plt


EPS = 1e-10


def mulRotT(vector, rot_matrix):
    return np.transpose(rot_matrix).dot(vector)


def mulRot(vector, rot_matrix):
    return rot_matrix.dot(vector)

# Rotation convention is: from world to special: use mulRotT, from special to world: use mulRot
# Exception is contact frame rotation, which is inverted.


def weighted_sum_vectors(vector1, vector2, weight1, weight2):
    return (vector1 * weight1 + vector2 * weight2) / (weight1 + weight2)


def normalize_vectors(vectors):
    mag = np.linalg.norm(vectors, axis=1, ord=2)
    return vectors / np.expand_dims(mag, -1)


# ======================== Mujoco access utils ====================================
# =================================================================================


def get_geom_id(mujoco_model, geom_id=None, geom_name=None):
    """ Allows access to mujoco geoms using either the id or the name of the geom """
    if geom_id is None and geom_name is None:
        raise RuntimeError("Need either name or geom id")

    if geom_id is None:
        geom_id = mujoco_model.geom_name2id(geom_name)

    return geom_id


def get_body_id(mujoco_model, body_id=None, body_name=None):
    """ Allows access to mujoco bodies using either the id or the name of the geom """
    if body_id is None and body_name is None:
        raise RuntimeError("Need either name or body id")

    if body_id is None:
        body_id = mujoco_model.body_name2id(body_name)

    return body_id


# ======================== Mujoco frame utils =====================================
# =================================================================================


def get_geoms_for_body(sim_model, body_id):
    geom_start = sim_model.body_geomadr[body_id]
    geom_end = geom_start + sim_model.body_geomnum[body_id]
    return range(geom_start, geom_end)


def get_geom_position(sim_data, geom_id):
    """ Returns world position of geom"""
    return sim_data.geom_xpos[geom_id]


def get_body_position(sim_data, body_id):
    """ Returns world position of body"""
    return sim_data.body_xpos[body_id]


def get_geom_rotation(sim_data, geom_id):
    """ Returns rotation matrix of geom frame relative to world frame"""
    return np.reshape(sim_data.geom_xmat[geom_id], (3, 3))


def get_body_rotation(sim_data, body_id):
    """ Returns rotation matrix of geom frame relative to world frame"""
    return np.reshape(sim_data.body_xmat[body_id], (3, 3))


def world_pos_to_geom(sim_data, position, geom_id):
    """ Converts a (n, 3) numpy array containing xyz coordinates in world frame to geom frame"""
    rel_pos = position - get_geom_position(sim_data, geom_id)
    rel_pos = np.transpose(mulRotT(np.transpose(rel_pos), get_geom_rotation(sim_data, geom_id)))
    return rel_pos


def world_pos_to_body(sim_data, position, body_id):
    """ Converts a (n, 3) numpy array containing xyz coordinates in world frame to body frame"""
    rel_pos = position - get_body_position(sim_data, body_id)
    rel_pos = world_rot_to_body(sim_data, rel_pos, body_id)
    return rel_pos


def geom_pos_to_world(sim_data, position, geom_id):
    """ Converts a (n, 3) numpy array containing xyz coordinates in geom frame to world frame"""
    global_pos = geom_rot_to_world(sim_data, position, geom_id)
    global_pos = global_pos + get_geom_position(sim_data, geom_id)
    return global_pos


def body_pos_to_world(sim_data, position, body_id):
    """ Converts a (n, 3) numpy array containing xyz coordinates from body frame to world frame"""
    global_pos = body_rot_to_world(sim_data, position, body_id)
    global_pos = global_pos + get_body_position(sim_data, body_id)
    return global_pos


def geom_pos_to_body(sim_data, position, geom_id, body_id):
    world_pos = geom_pos_to_world(sim_data, position, geom_id)
    return world_pos_to_body(sim_data, world_pos, body_id)


def geom_rot_to_world(sim_data, vector, geom_id):
    return np.transpose(mulRot(np.transpose(vector), get_geom_rotation(sim_data, geom_id)))


def body_rot_to_world(sim_data, vector, body_id):
    return np.transpose(mulRot(np.transpose(vector), get_body_rotation(sim_data, body_id)))


def world_rot_to_geom(sim_data, vector, geom_id):
    return np.transpose(mulRotT(np.transpose(vector), get_geom_rotation(sim_data, geom_id)))


def world_rot_to_body(sim_data, vector, body_id):
    return np.transpose(mulRotT(np.transpose(vector), get_body_rotation(sim_data, body_id)))


def geom_rot_to_body(sim_data, vector, geom_id, body_id):
    world_rot = geom_rot_to_world(sim_data, vector, geom_id)
    return world_rot_to_body(sim_data, world_rot, body_id)


# ======================== Mujoco data utils ======================================
# =================================================================================


def get_data_for_sensor(sim, sensor_name):
    """ Get sensor data for sensor sensor_name"""
    sensor_id = sim.model.sensor_name2id(sensor_name)
    start = sim.model.sensor_adr[sensor_id]
    end = start + sim.model.sensor_dim[sensor_id]
    return sim.data.sensordata[start:end]


def _decode_name(sim, name_adr):
    """ Mujoco-py unfortunately does not properly wrap all of mujocos data structures/functions, so we have to get some
    names (such as textures and materials) manually. This is a very tedious process in python """
    # TODO: Figure out cython so we don't have to do this
    # TODO: Alternatively at least cache the name-id relationship somewhere
    i = 0
    while sim.model.names[name_adr + i].decode() != "":
        i += 1
    if i == 0:
        return None
    str_array = sim.model.names[name_adr: name_adr + i].astype(str)
    return "".join(str_array)


def texture_name2id(sim, texture_name):
    """ Returns the id for the texture with the given name. """
    tex_id = None
    for i, name_adr in enumerate(sim.model.name_texadr):
        name = _decode_name(sim, name_adr)
        if name == texture_name:
            tex_id = i
            break
    if tex_id is None:
        raise RuntimeError("Could not find texture with name {}".format(texture_name))
    return tex_id


def material_name2id(sim, material_name):
    """ Returns the id for the material with the given name. """
    mat_id = None
    for i, name_adr in enumerate(sim.model.name_matadr):
        name = _decode_name(sim, name_adr)
        if name == material_name:
            mat_id = i
            break
    if mat_id is None:
        raise RuntimeError("Could not find material with name {}".format(material_name))
    return mat_id


# ======================== Plotting utils =========================================
# =================================================================================

def plot_points(points, limit: float = 1.0, title=""):
    xs = points[:, 0]
    ys = points[:, 1]
    zs = points[:, 2]
    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs, ys, zs, color="k", s=20)
    ax.set_title(title)
    ax.set_xlim([-limit, limit])
    ax.set_ylim([-limit, limit])
    ax.set_zlim([-limit, limit])
    ax.set_box_aspect((1, 1, 1))
    plt.tight_layout()
    plt.show()


def plot_forces(points, vectors, limit: float = 1.0, title=""):
    xs = points[:, 0]
    ys = points[:, 1]
    zs = points[:, 2]

    us = vectors[:, 0]
    vs = vectors[:, 1]
    ws = vectors[:, 2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.quiver(xs, ys, zs, us, vs, ws)
    ax.scatter(xs, ys, zs, color="k", s=20)
    ax.set_title(title)
    ax.set_xlim([-limit, limit])
    ax.set_ylim([-limit, limit])
    ax.set_zlim([-limit, limit])
    ax.set_box_aspect((1, 1, 1))
    plt.tight_layout()
    plt.show()
