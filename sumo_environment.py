import traci
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import random
import time
import subprocess
import os
import json

from config import CONFIG


SCREEN_WIDTH = 1400
SCREEN_HEIGHT = 800


class SUMOEnvironment:
    """Entorno para RL en SUMO con un vehículo controlable"""
    DEFAULT_REWARD_WEIGHTS = {
        "collision": -100.0,         # Penalización por colisión
        "teleport": -50.0,           # Penalización por teleportación (indica un bloqueo grave)
        "emergency_stop": -10.0,     # Penalización por parada de emergencia
        "intersection_stop": -1.0,   # Penalización menor en intersecciones para evitar quedarse parado
        "lane_change_fail": -5.0,    # Penalización por fallo en cambio de carril
        "route_error": -10.0,        # Penalización por error en la ruta
        "speed_reward": 0.1,         # Recompensa por velocidad (por m/s)
        "step_reward": -0.01,        # Pequeña penalización por cada paso (para incentivar completar rápido)
        "goal_reached": 200.0,       # Recompensa por alcanzar el final de un tramo
        "wrong_direction": -15.0,    # Penalización por ir en dirección contraria
        "keep_lane": 0.05,           # Pequeña recompensa por mantener el carril (evitar zigzag)
        "approaching_goal": 3,       # Recompensa por reducir distancia al final del edge actual
        "ran_red_light": -15.0,      # Penalización alta por pasar en rojo
        "stopped_at_red": 0.5,       # Recompensa pequeña por detenerse correctamente
        "passed_green_light": 0.2    # Recompensa muy pequeña por pasar en verde
    }
    ACTION_NAMES = {
        0: "No hacer nada (mantener velocidad y dirección)",
        1: "Acelerar",
        2: "Frenar / Marcha atras",
        3: "Cambiar al carril izquierdo",
        4: "Cambiar al carril derecho",
        5: "Girar a la izquierda en la próxima intersección",
        6: "Girar a la derecha en la próxima intersección",
        7: "Seguir recto en la próxima intersección",
    }
    MAX_EXPECTED_SPEED = 33.33 # m/s
    DEFAULT_PORT = 8813
    MAX_MAP_COORD_X = 2000.0
    MAX_MAP_COORD_Y = 2500.0
    MAX_EDGE_LEN = 500.0
    
    def __init__(
        self, 
        sumo_config: str,
        vehicle_id: str = "manualVehicle",
        vehicle_type: str = "veh_passenger",
        route_type_id: str = "passenger",
        max_steps: int = 6000,
        gui: bool = True,
        alg_name: str = None,
        detection_radius: float = 50.0,
        reward_weights: Dict[str, float] = None,
        STATE_DIM: int = 59,
        port: Optional[int] = None,
        eval_route: list[str] = []
    ):
        self.sumo_config = sumo_config
        self.vehicle_id = vehicle_id
        self.vehicle_type = vehicle_type
        self.route_type_id = route_type_id
        self.max_steps = max_steps
        self.gui = gui
        self.alg_name = alg_name
        self.detection_radius = detection_radius
        self.STATE_DIM = STATE_DIM
        self.port = port
        self.eval_route = eval_route
        self.current_edge_goal = 0
        
        # Pesos para la función de recompensa
        self.reward_weights = reward_weights or self.DEFAULT_REWARD_WEIGHTS
        
        # Estado actual
        self.current_step = 0
        self.vehicle_active = False
        self.episode_reward = 0.0
        self.collision_count = 0
        self.teleport_count = 0
        self.emergency_stops = 0
        self.last_position = None
        self.total_distance = 0.0
        self.last_lane = None
        self.last_action = None
        self.vehicle_stats = {}
        self.forced_collision = False
        self.last_dist_to_end = None
        
        # Variable para controlar si la simulación ya está iniciada
        self.simulation_running = False

        # Punto de interes para resaltar el objetivo en modo gui
        self.TARGET_MARKER_ID_POI = "target_marker_poi"
        
        # Inicializar la simulación
        self._start_simulation()
        
    def _start_simulation(self):
        """Inicia la simulación de SUMO"""
        # Verificar si SUMO ya está en ejecución y cerrarlo si es necesario
        if self.simulation_running:
            try:
                traci.close()
                # Esperar un momento para asegurar que se cierre correctamente
                time.sleep(0.5)
            except:
                pass
            self.simulation_running = False
        
        traci_port = self.port if self.port is not None else self.DEFAULT_PORT

        # Comando para iniciar SUMO
        sumo_cmd = [
            "sumo-gui" if self.gui else "sumo", 
            "-c", self.sumo_config, 
            "--ignore-route-errors", "true",
            "--collision.action", "warn",  # Para detectar colisiones sin detener la simulación
            "--collision.stoptime", "0",
            "--step-length", "0.1",        # Pasos de simulación más pequeños (0.1 segundos)
            "--no-warnings", "true",
            "--start",                     # Iniciar la simulación inmediatamente
            "--quit-on-end",               # Cerrar SUMO cuando termine la simulación
            # "--remote-port", str(traci_port)
        ]
        if self.gui:
            sumo_cmd.extend([
                "--window-size", str(SCREEN_WIDTH) + "," + str(SCREEN_HEIGHT), # Establecer tamaño de ventana
                "--window-pos", "0,0",          # Posiciona la ventana en la esquina superior izquierda
                # "--maximized", "true",
            ])
        
        # Intentar iniciar SUMO con varios reintentos
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                if not traci.isLoaded():
                    traci.start(sumo_cmd, port=traci_port)
                    self.simulation_running = True
                    # if self.gui:
                    #     traci.gui.setSchema(traci.gui.DEFAULT_VIEW, "fullscreen")
                    print(f"SUMO iniciado y TraCI conectado en puerto {traci_port}.")
                break
            except Exception as e:
                print(f"Intento {attempt+1}/{max_attempts} de iniciar SUMO falló (Puerto {traci_port}): {e}")
                time.sleep(3)  # Esperar antes de reintentar
        
        if not self.simulation_running:
            raise RuntimeError(f"No se pudo iniciar la simulación de SUMO después de varios intentos (Puerto {traci_port})")
        
    def reset(self, eval_route: list[str] = []) -> np.ndarray:
        """Reinicia el entorno y devuelve el estado inicial"""
        # Cerrar la simulación si está en ejecución
        if self.simulation_running:
            try:
                if traci.isLoaded():
                    traci.close()
                # Esperar un momento para asegurar que se cierre correctamente
                time.sleep(0.5)
            except:
                pass
            self.simulation_running = False
            
        # Reiniciar la simulación
        self._start_simulation()
        
        # Reiniciar variables de estado
        self.current_step = 0
        self.vehicle_active = False
        self.episode_reward = 0.0
        self.collision_count = 0
        self.teleport_count = 0
        self.emergency_stops = 0
        self.last_position = None
        self.total_distance = 0.0
        self.last_lane = None
        self.last_action = None
        self.forced_collision = False
        self.last_dist_to_end = None
        self.eval_route = eval_route
        self.current_edge_goal = 0

        
        # Añadir vehículo a la simulación
        try:
            self._add_vehicle()
            self.vehicle_active = True
            print("Vehículo añadido con éxito")
        except Exception as e:
            print(f"Error al inicializar vehículo: {e}")
            
        # Avanzar un paso para que el vehículo entre en la simulación
        if traci.isLoaded():
            traci.simulationStep()
        else:
            print("TraCI no está conectado después de resetear el entorno")
            self._start_simulation()
            traci.simulationStep()
        
        # Obtener el estado inicial
        state = self._get_state()
        return state
        
    def _add_vehicle(self):
        """Añade el vehículo a la simulación"""
        if not traci.isLoaded():
            raise RuntimeError("TraCI no está conectado, no se puede añadir vehículo")
            
        if self.eval_route:
            route = self.eval_route
            start_edge = route[self.current_edge_goal]
            self.current_edge_goal += 1
            dest_edge = route[self.current_edge_goal]
            self._add_poi(dest_edge)
        else:
            # Obtener edges válidos
            edges = traci.edge.getIDList()
            
            # Verificar que hay edges disponibles
            if not edges:
                raise ValueError("No hay edges disponibles en la simulación")

            # Encontrar edge de inicio aleatorio
            start_edge = self._find_random_start_edge(edges)
            if start_edge is None:
                raise ValueError("No se pudo encontrar un edge de inicio aleatorio válido.")

            # Encontrar edge de destino aleatorio desde el inicio
            dest_edge = self._find_random_next_edge(start_edge, edges)
            if dest_edge is None:
                raise ValueError(f"No se pudo encontrar un destino válido ni siquiera desde {start_edge}.")

        # Crear la ruta inicial
        route = [start_edge, dest_edge]
        route_id = f"{self.vehicle_id}_route_{random.randint(1000, 9999)}"
        try:
            # Añadir la nueva ruta (eliminar si existe una con el mismo ID es buena práctica, aunque poco probable aquí)
            if route_id in traci.route.getIDList():
                traci.route.remove(route_id)
            traci.route.add(route_id, route)
            print(f"Ruta inicial creada: {route_id} = {route}")

            # Eliminar vehículo anterior si aún existe (importante en reset)
            if self.vehicle_id in traci.vehicle.getIDList():
                print(f"Eliminando vehículo existente: {self.vehicle_id}")
                traci.vehicle.remove(self.vehicle_id)
                # Dar un pequeño respiro a TraCI
                traci.simulationStep() # Avanzar un paso para procesar la eliminación
            
            # Añadir vehículo
            traci.vehicle.add(
                vehID=self.vehicle_id,
                routeID=route_id,
                typeID=self.vehicle_type,
                depart="now",        # Empezar inmediatamente
                departLane="random", # Empezar en un carril aleatorio del start_edge
                departPos="random",  # Empezar en una posición aleatoria del carril
                departSpeed="0"      # Empezar parado o con velocidad aleatoria "random" / "max"
            )
            print(f"Vehículo '{self.vehicle_id}' añadido en edge '{start_edge}' con ruta a '{dest_edge}'.")

            # Esperar a que el vehículo realmente entre en la simulación
            step = 0
            max_wait_steps = 50 # Esperar máximo 5 segundos
            while self.vehicle_id not in traci.vehicle.getIDList() and step < max_wait_steps:
                traci.simulationStep()
                step += 1

            if self.vehicle_id not in traci.vehicle.getIDList():
                raise RuntimeError(f"El vehículo {self.vehicle_id} no apareció en la simulación después de {max_wait_steps} pasos.")

            # Configuraciones del vehículo (color, cámara, control)
            traci.vehicle.setColor(self.vehicle_id, (7, 246, 233, 255)) # Cambiar color

            # Fijar camara en el vehiculo insertado
            if self.gui:
                traci.gui.trackVehicle("View #0", self.vehicle_id)
                traci.gui.setZoom("View #0", 2000)
            
            # Configurar propiedades del vehículo
            traci.vehicle.setParameter(self.vehicle_id, "ignore-route-errors", "true")
            traci.vehicle.setLaneChangeMode(self.vehicle_id, 0)  # Control total sobre cambios de carril
            traci.vehicle.setSpeedMode(self.vehicle_id, 0)       # Control total sobre velocidad
            return True

        except Exception as e:
            print(f"Error en _add_vehicle: {e}")
            # Intentar limpiar si falló
            if route_id in traci.route.getIDList():
                traci.route.remove(route_id)
            if self.vehicle_id in traci.vehicle.getIDList():
                traci.vehicle.remove(self.vehicle_id)
            return False

    def _add_poi(self, edge):
        if not self.gui:
            return

        if self.TARGET_MARKER_ID_POI in traci.poi.getIDList():
            traci.poi.remove(self.TARGET_MARKER_ID_POI)


        # Calcular posición
        try:
            # Tomar el primer carril del edge como referencia
            next_lane_id = f"{edge}_0"
            lane_shape = traci.lane.getShape(next_lane_id)
            if lane_shape:
                target_pos = lane_shape[0] # Posición inicial del carril
            else:
                    print(f"Advertencia: No se pudo obtener la forma del carril {next_lane_id} para POI.")
        except traci.exceptions.TraCIException as e_poi_pos:
                print(f"Error obteniendo posición para POI en {edge}: {e_poi_pos}")

        if target_pos:
            traci.poi.add(
                poiID=self.TARGET_MARKER_ID_POI,
                x=target_pos[0],
                y=target_pos[1],
                color=(255, 0, 0, 255), # Rojo
                layer=10
            )

    def _get_all_possible_next_edge(self, current_edge_id: str, edges: List[str]) -> Optional[List[str]]:
        """
        Dado un edge actual, encuentra todos los edge de destino válidos
        conectado a él que permita el tipo de vehículo del agente.
        """
        if not traci.isLoaded() or current_edge_id not in edges:
            return None
        
        possible_next_edges = []
        try:
            num_lanes_start = traci.edge.getLaneNumber(current_edge_id)
            lanes_allowing_vehicle = []
            # Encontrar carriles en el edge actual que permitan el vehículo
            for i in range(num_lanes_start):
                lane_id = f"{current_edge_id}_{i}"
                allowed_vehicles = traci.lane.getAllowed(lane_id)
                if not allowed_vehicles or self.vehicle_type in allowed_vehicles or self.route_type_id in allowed_vehicles:
                    lanes_allowing_vehicle.append(lane_id)

            # Buscar salidas válidas desde esos carriles
            for lane_id in lanes_allowing_vehicle:
                links = traci.lane.getLinks(lane_id)
                for link in links:
                    next_lane_id = link[0]
                    if ":" not in next_lane_id: # Ignorar conexiones internas
                        next_edge_id = traci.lane.getEdgeID(next_lane_id)
                        # Verificar que sea diferente, exista y permita el vehículo
                        if next_edge_id != current_edge_id and next_edge_id in edges:
                            next_edge_allows = False
                            try:
                                for j in range(traci.edge.getLaneNumber(next_edge_id)):
                                    next_lane_check = f"{next_edge_id}_{j}"
                                    allowed_vehicles = traci.lane.getAllowed(next_lane_check)
                                    if not allowed_vehicles or self.vehicle_type in allowed_vehicles or self.route_type_id in allowed_vehicles:
                                        next_edge_allows = True
                                        break
                                if next_edge_allows:
                                    possible_next_edges.append(next_edge_id)
                            except traci.exceptions.TraCIException:
                                # Ignorar errores al verificar edge de destino (puede ser inválido temporalmente)
                                continue
            
            possible_next_edges = list(set(possible_next_edges)) # Eliminar duplicados
            return possible_next_edges

        except traci.exceptions.TraCIException as e:
            print(f"Error TraCI buscando salidas para {current_edge_id}: {e}")
            return None
        except Exception as e:
            print(f"Error inesperado buscando salidas para {current_edge_id}: {e}")
            return None

    
    def _find_random_next_edge(self, current_edge_id: str, edges: List[str]) -> Optional[str]:
        """
        Dado un edge actual, encuentra un edge de destino aleatorio válido
        conectado a él que permita el tipo de vehículo del agente.
        """
        if not traci.isLoaded() or current_edge_id not in edges:
            return None

        possible_next_edges = []
        try:
            possible_next_edges = self._get_all_possible_next_edge(current_edge_id, edges)

            if possible_next_edges:
                edge = random.choice(possible_next_edges)
                self._add_poi(edge)
                return edge
            else:
                print(f"Advertencia: No se encontraron salidas válidas desde {current_edge_id}")
                return None

        except traci.exceptions.TraCIException as e:
            print(f"Error TraCI buscando salida aleatoria para {current_edge_id}: {e}")
            return None
        except Exception as e:
            print(f"Error inesperado buscando salida aleatoria para {current_edge_id}: {e}")
            return None

    def _find_random_start_edge(self, edges: List[str]) -> Optional[str]:
        """
        Encuentra un edge de inicio aleatorio válido que permita el vehículo
        y que tenga al menos una salida válida a otro edge permitido.
        """
        possible_starts = []
        print(f"Buscando edges de inicio válidos para tipo '{self.vehicle_type}'...")
        for edge_id in edges:
            if edge_id.startswith(":"): # Ignorar edges internos
                continue
            try:
                # Verificar permiso del vehículo (rápido)
                num_lanes = traci.edge.getLaneNumber(edge_id)
                edge_allows_vehicle = False
                for i in range(num_lanes):
                    lane_id = f"{edge_id}_{i}"
                    allowed_vehicles = traci.lane.getAllowed(lane_id)
                    if not allowed_vehicles or self.vehicle_type in allowed_vehicles or self.route_type_id in allowed_vehicles:
                        edge_allows_vehicle = True
                        break
                if not edge_allows_vehicle:
                    continue

                # Verificar si tiene al menos una salida válida usando la función genérica
                if len(self._get_all_possible_next_edge(edge_id, edges)) > 0:
                    possible_starts.append(edge_id)

            except traci.exceptions.TraCIException:
                pass # Ignorar edges inválidos
            except Exception as e:
                print(f"Error inesperado procesando edge {edge_id}: {e}")

        if not possible_starts:
            print("Error: No se encontraron edges de inicio válidos en el mapa.")
            return None
        else:
            print(f"Encontrados {len(possible_starts)} posibles edges de inicio.")
            return random.choice(possible_starts)
    
    # TODO: revisar metodos obsoletos
    def _get_valid_route(self, edges: List[str]) -> list[str]:
        valid_edges = []
        for edge in edges:
            # Obtener carriles del edge
            num_lanes = traci.edge.getLaneNumber(edge)
            lanes = [f"{edge}_{i}" for i in range(num_lanes)]
            for lane in lanes:
                # Obtener vehiculos permitidos en el carril
                allowed_vehicles = traci.lane.getAllowed(lane)

                # Si la lista está vacía, significa que todos los vehículos están permitidos
                # O si el tipo de vehículo está explícitamente permitido
                if not allowed_vehicles or self.vehicle_type in allowed_vehicles or self.route_type_id in allowed_vehicles:
                    # Obtener conexiones del carril
                    connections = traci.lane.getLinks(lane)
                    for connection in connections:
                        if connection[1] is True:
                            valid_edges.append(edge)

        # Si no hay edges válidos, lanzar error
        if not valid_edges:
            raise ValueError("No se encontraron edges válidos para la ruta")
        
        # Simplificamos la lógica para encontrar una ruta: intentamos obtener la longitud de otra manera
        # Usamos una función alternativa dependiendo de la versión de SUMO
        try:
            # Intentar varios métodos posibles para obtener la longitud
            edge_lengths = {}
            for edge in valid_edges:
                try:
                    # Opción 1: Método directo (versiones más recientes)
                    edge_lengths[edge] = traci.edge.getLength(edge)
                except AttributeError:
                    try:
                        # Opción 2: A través del primer carril del edge
                        lane_id = f"{edge}_0"  # Primer carril
                        edge_lengths[edge] = traci.lane.getLength(lane_id)
                    except:
                        # Opción 3: Si no podemos obtener la longitud, asignamos un valor por defecto
                        edge_lengths[edge] = 0
            
            # Ordenar los edges por longitud
            sorted_edges = sorted(valid_edges, key=lambda edge: edge_lengths[edge], reverse=True)
            # sorted_edges = sorted(valid_edges, key=lambda edge: edge_lengths[edge], reverse=False)
            
        except Exception as e:
            # Si hay algún error, simplemente devolvemos el primer edge válido
            print(f"Error al ordenar edges por longitud: {e}")
            sorted_edges = valid_edges
        
        # Devolver al menos un edge válido (el más largo si fue posible ordenarlos)
        return [sorted_edges[0]]
        
    def _assign_new_route(self) -> Optional[List[str]] | dict:
        """
        Asigna una nueva ruta al vehículo desde su edge actual.
        """
        if not self.vehicle_active or self.vehicle_id not in traci.vehicle.getIDList():
            print("Error en _assign_new_route: Vehículo no activo.")
            return None

        if self.eval_route:
            if self.current_edge_goal < len(self.eval_route):
                new_route_edges = self.eval_route[self.current_edge_goal: self.current_edge_goal+2]
                self.current_edge_goal += 1
                self._add_poi(new_route_edges[-1])
                return new_route_edges
            else:
                print("¡Final de la ruta de evualuación alzancado!")
                return {"success": True}
                

        try:
            current_edge = traci.vehicle.getRoadID(self.vehicle_id)
            # Si está en una intersección, no podemos asignar ruta desde ahí directamente
            if ":" in current_edge:
                print("Advertencia: Intentando asignar nueva ruta desde intersección. Esperando al siguiente edge.")
                # Podríamos intentar obtener el *siguiente* edge de la ruta actual, pero
                # es más simple dejar que el vehículo salga de la intersección primero.
                # Devolver None para que no se cambie la ruta aún.
                return None # No hacer nada si está en intersección

            edges = traci.edge.getIDList() # Obtener lista actualizada

            # Encontrar un destino aleatorio desde el edge actual
            next_edge = self._find_random_next_edge(current_edge, edges)

            if next_edge is not None:
                new_route_edges = [current_edge, next_edge]
                traci.vehicle.setRoute(self.vehicle_id, new_route_edges)
                print(f"Nueva ruta asignada desde '{current_edge}' -> '{next_edge}'")
                return new_route_edges
            else:
                print(f"Advertencia: No se encontró un siguiente edge válido desde '{current_edge}'. El vehículo se detendrá o desaparecerá.")
                # Podríamos forzar una colisión aquí o dejar que SUMO lo maneje
                # self._force_collision(action=-1) # Usar una acción dummy
                return None

        except traci.exceptions.TraCIException as e:
            print(f"Error TraCI al asignar nueva ruta: {e}")
            return None
        except Exception as e:
            print(f"Error inesperado al asignar nueva ruta: {e}")
            return None
        
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Ejecuta un paso con la acción dada.
        Actions:
        0: No hacer nada (mantener velocidad y dirección)
        1: Acelerar
        2: Frenar / Marcha atras
        3: Cambiar al carril izquierdo
        4: Cambiar al carril derecho
        5: Girar a la izquierda en la próxima intersección
        6: Girar a la derecha en la próxima intersección
        7: Seguir recto en la próxima intersección
        """
        if not self.simulation_running or not traci.isLoaded():
            print("La simulación no está en ejecución o TraCI no está conectado")
            return np.zeros(self.STATE_DIM), self.reward_weights["collision"], True, {"terminal": "simulation_closed"}
            
        self.last_action = action
        reward = 0.0
        done = False
        info = {}
        if self.gui:
            self._save_frame()
        
        # Comprobar si el vehículo existe
        if not self.vehicle_active or self.vehicle_id not in traci.vehicle.getIDList():
            print(f"Vehículo '{self.vehicle_id}' no está en la simulación. Aplicando penalización.")
            self.vehicle_active = False
            return self._get_state(), self.reward_weights["collision"], True, {"terminal": "vehicle_removed"}
        
        # Guardar estado anterior para calcular recompensas
        prev_position = None
        prev_lane = None
        try:
            prev_position = traci.vehicle.getPosition(self.vehicle_id)
            prev_lane = traci.vehicle.getLaneID(self.vehicle_id)
        except:
            pass
            
        # Aplicar acción
        try:
            self._apply_action(action)
        except traci.exceptions.TraCIException as e:
            print(f"Error al aplicar acción: {e}")
            reward += self.reward_weights["route_error"]
        
        # Avanzar simulación
        if not traci.isLoaded():
            return self._get_state(), self.reward_weights["collision"], True, {"terminal": "simulation_closed"}
        
        try:
            traci.simulationStep()
            self.current_step += 1
        except traci.exceptions.FatalTraCIError as e:
            print(f"Error fatal en TraCI durante simulationStep: {e}")
            return self._get_state(), self.reward_weights["collision"], True, {"terminal": "traci_error"}
        
        # Comprobar si el vehículo sigue activo
        vehicle_exists = self.vehicle_id in traci.vehicle.getIDList()
        
        # Calcular recompensa y actualizar estadísticas
        step_reward, info = self._calculate_reward(prev_position, prev_lane, vehicle_exists)
        reward += step_reward
        self.episode_reward += reward
        
        # Comprobar fin de episodio
        done = (
            not vehicle_exists or 
            self.current_step >= self.max_steps or
            ("terminal" in info and info.get("terminal") != "new_route_assigned")
        )
        
        # Reiniciar vehículo si es necesario y no ha terminado el episodio
        if not vehicle_exists and not done:
            try:
                success = self._add_vehicle()
                self.vehicle_active = success
                if not success:
                    done = True
            except Exception as e:
                print(f"No se pudo reinsertar el vehículo: {e}")
                self.vehicle_active = False
                done = True
        
        # Obtener estado actual
        state = self._get_state()
        
        # Actualizar información adicional
        info.update({
            "step": self.current_step,
            "episode_reward": self.episode_reward,
            "collision_count": self.collision_count,
            "teleport_count": self.teleport_count,
            "total_distance": self.total_distance,
        })
        
        return state, reward, done, info
        
    def _apply_action(self, action: int):
        """Aplica la acción seleccionada al vehículo"""
        if not self.vehicle_active or self.vehicle_id not in traci.vehicle.getIDList():
            return
            
        # Obtener velocidad y carril actuales
        current_speed = traci.vehicle.getSpeed(self.vehicle_id)
        current_lane_index = traci.vehicle.getLaneIndex(self.vehicle_id)
        max_speed = traci.vehicle.getMaxSpeed(self.vehicle_id)
        
        # Aplicar acción
        if action == 0:  # Mantener velocidad y dirección
            pass
        
        elif action == 1:  # Acelerar
            target_speed = min(current_speed + 2.0, max_speed)
            traci.vehicle.setSpeed(self.vehicle_id, target_speed)
            
        elif action == 2:  # Frenar / Marcha atras
            target_speed = max(current_speed - 3.0, -max_speed)
            traci.vehicle.setSpeed(self.vehicle_id, target_speed)
            
        elif action == 3:  # Cambiar al carril izquierdo
            try:
                traci.vehicle.changeLane(self.vehicle_id, current_lane_index + 1, 3.0)
            except Exception as e:
                self._force_collision(action)
            
        elif action == 4:  # Cambiar al carril derecho
            # if current_lane_index > 0:
            try:
                traci.vehicle.changeLane(self.vehicle_id, current_lane_index - 1, 3.0)
            except Exception as e:
                self._force_collision(action)
                
        elif action in [5, 6, 7]:  # Maniobras en intersección
            current_lane = traci.vehicle.getLaneID(self.vehicle_id)
            links = traci.lane.getLinks(current_lane)
            # Forzamos la maniobra, incluso si no hay enlaces válidos
            if links:
                if action == 5:  # Girar a la izquierda
                    direction_links = [link for link in links if link[6] == 'l']
                elif action == 6:  # Girar a la derecha
                    direction_links = [link for link in links if link[6] == 'r']
                elif action == 7:  # Seguir recto
                    direction_links = [link for link in links if link[6] == 's']

                # Si no hay enlace que cumpla la condición, forzamos una colision
                if direction_links:
                    # Tomar el primer link disponible con la dirección deseada
                    next_lane = direction_links[0][0]
                    next_edge = traci.lane.getEdgeID(next_lane)
                    current_edge = traci.vehicle.getRoadID(self.vehicle_id)
                    traci.vehicle.setRoute(self.vehicle_id, [current_edge, next_edge])
                else:
                    self._force_collision(action)
            else:
                # Si no hay enlaces en absoluto, forzamos colision también
                self._force_collision(action)

    def _force_collision(self, action: int):
        """
        Fuerza una colisión cuando se intenta una acción inválida. Esto puede ocurrir cuando:
        - Se intenta cambiar a un carril que no existe
        - Se intenta tomar una dirección no disponible en una intersección
        """
        self.forced_collision = True
        
        # Eliminar el vehículo de la simulación para simular la colisión
        traci.vehicle.remove(self.vehicle_id, reason=4)  # reason=4 indica colisión
        
        # Marcar el vehículo como inactivo
        self.vehicle_active = False
        
        # Reportar la acción que causó la colisión
        action_name = self.ACTION_NAMES.get(action, str(action))
        print(f"Colisión forzada por acción inválida: {action_name}")
            
    
    def _get_state(self) -> np.ndarray:
        """
        Obtiene el estado actual del vehículo y su entorno.
        """
        if not self.simulation_running or not traci.isLoaded() or not self.vehicle_active or self.vehicle_id not in traci.vehicle.getIDList():
            # Devolver un estado de "dummy" con ceros
            return np.zeros(self.STATE_DIM)
            
        # 1. Información básica del vehículo
        position = traci.vehicle.getPosition(self.vehicle_id)
        speed = traci.vehicle.getSpeed(self.vehicle_id)
        angle = traci.vehicle.getAngle(self.vehicle_id)
        
        # Convertir ángulo a radianes y a componentes seno y coseno
        angle_rad = np.radians(angle)
        angle_sin = np.sin(angle_rad)
        angle_cos = np.cos(angle_rad)

        # Componentes de velocidad aproximados usando velocidad escalar y dirección
        speed_x = speed * angle_cos
        speed_y = speed * angle_sin
        speed_vector = [speed_x, speed_y, 0.0]  # Añadimos componente z = 0
        
        # 2. Información del carril y edge
        lane_id = traci.vehicle.getLaneID(self.vehicle_id)
        lane_index = traci.vehicle.getLaneIndex(self.vehicle_id)
        edge_id = traci.vehicle.getRoadID(self.vehicle_id)
        
        # Si estamos en una intersección, algunos datos no están disponibles
        in_intersection = ":" in lane_id
        if not in_intersection and edge_id != "":
            lane_count = traci.edge.getLaneNumber(edge_id)
            edge_speed_limit = traci.lane.getMaxSpeed(lane_id)
        else:
            lane_count = 1
            edge_speed_limit = 13.9  # ~50 km/h como valor predeterminado
        
        # 3. Detección de vehículos cercanos
        nearby_vehicles = []
        
        # Detectar vehículos cercanos en un radio
        vehicles_in_radius = self._get_nearby_vehicles(position, self.detection_radius)
        
        # Para cada vehículo, calcular distancia relativa y velocidad relativa
        for v_id, v_data in vehicles_in_radius.items():
            if v_id != self.vehicle_id:
                rel_x = v_data['position'][0] - position[0]
                rel_y = v_data['position'][1] - position[1]


                # Calcular velocidades relativas usando las velocidades aproximadas
                v_speed = v_data['speed']
                v_angle = v_data['angle']
                v_angle_rad = np.radians(v_angle)
                v_speed_x = v_speed * np.cos(v_angle_rad)
                v_speed_y = v_speed * np.sin(v_angle_rad)

                rel_speed_x = v_speed_x - speed_vector[0]
                rel_speed_y = v_speed_y - speed_vector[0]

                distance = np.sqrt(rel_x**2 + rel_y**2)
                
                # Normalizar valores según la dirección del vehículo
                rel_x_norm = rel_x * angle_cos + rel_y * angle_sin
                rel_y_norm = -rel_x * angle_sin + rel_y * angle_cos
                
                nearby_vehicles.append([
                    rel_x_norm / self.detection_radius,
                    rel_y_norm / self.detection_radius,
                    rel_speed_x / 30.0,
                    rel_speed_y / 30.0,
                    distance / self.detection_radius,
                ])
        
        # Asegurar que siempre tenemos el mismo número de vehículos en el estado (añadir placeholders)
        max_vehicles = 8
        while len(nearby_vehicles) < max_vehicles:
            nearby_vehicles.append([0, 0, 0, 0, 2.0])
        
        # Si hay más de 8 vehículos, quedarnos con los 8 más cercanos
        if len(nearby_vehicles) > max_vehicles:
            nearby_vehicles = sorted(nearby_vehicles, key=lambda x: x[4])[:max_vehicles]
        
        # 4. Información de la próxima intersección
        next_links = [False, False, False]
        if not in_intersection and ":" not in lane_id:
            try:
                links = traci.lane.getLinks(lane_id)
                
                # Comprobar si hay enlaces (intersecciones cercanas)
                if links:
                    # Clasificar enlaces por dirección
                    directions = {'s': 0, 'l': 0, 'r': 0}
                    for link in links:
                        dir_type = link[6]  # s: recto, l: izquierda, r: derecha
                        if dir_type in directions:
                            directions[dir_type] += 1
                            
                    next_links = [
                        directions['s'] > 0,  # ¿Puedo ir recto?
                        directions['l'] > 0,  # ¿Puedo girar a la izquierda?
                        directions['r'] > 0   # ¿Puedo girar a la derecha?
                    ]
            except:
                next_links = [False, False, False]

        
        # 5. Información del Próximo Semáforo
        next_tls_state = [0.0, 0.0, 0.0] # One-hot: [Verde, Ambar, Rojo] por defecto (sin semáforo)
        dist_to_tls = 1.0 # Distancia normalizada (1.0 = muy lejos o sin semáforo)
        
        try:
            tls_info = traci.vehicle.getNextTLS(self.vehicle_id)
            
            if tls_info:
                # Tomar el más cercano
                tls_id, tls_index, tls_distance, tls_state = tls_info[0]
                dist_to_tls = min(tls_distance / self.detection_radius, 1.0) # Normalizar y limitar a 1

                # Codificar estado (simplificado)
                if tls_state.lower() == "g": # Verde o similar
                    next_tls_state = [1.0, 0.0, 0.0]
                elif tls_state.lower() == "y": # Ambar
                    next_tls_state = [0.0, 1.0, 0.0]
                elif tls_state.lower() == "r": # Rojo
                    next_tls_state = [0.0, 0.0, 1.0]
        except traci.exceptions.TraCIException:
            pass # Puede fallar si no hay semáforos en la ruta
        except Exception as e_tls:
            print(f"Error obteniendo info TLS: {e_tls}")

        # 6. Información del Edge Destino
        target_edge_rel_end_x = 0.0
        target_edge_rel_end_y = 0.0
        target_edge_norm_len = 0.0
        target_edge_id = None

        route = traci.vehicle.getRoute(self.vehicle_id)
        if route and len(route) > 1:
            target_edge_id = route[-1] # El último edge de la ruta actual (que es de 2 edges)
            target_lane_id = f"{target_edge_id}_0" # Usar el carril 0 como referencia
            target_shape = traci.lane.getShape(target_lane_id)
            target_len = traci.lane.getLength(target_lane_id)
            if target_shape:
                target_end_pos = target_shape[-1] # Coordenadas absolutas del final del carril
                # Calcular posición relativa al vehículo actual
                rel_x = target_end_pos[0] - position[0]
                rel_y = target_end_pos[1] - position[1]
                # Rotar al marco de referencia del vehículo
                target_edge_rel_end_x = (rel_x * angle_cos + rel_y * angle_sin) / self.MAX_MAP_COORD_X
                target_edge_rel_end_y = (-rel_x * angle_sin + rel_y * angle_cos) / self.MAX_MAP_COORD_Y
            target_edge_norm_len = target_len / self.MAX_EDGE_LEN
        
        # Compilar todo el estado
        state_components = [
            position[0] / self.MAX_MAP_COORD_X,
            position[1] / self.MAX_MAP_COORD_Y,
            speed / self.MAX_EXPECTED_SPEED,
            angle_sin,
            angle_cos,
            lane_index / max(lane_count, 1),
            lane_count / 5.0,
            edge_speed_limit / self.MAX_EXPECTED_SPEED,
            int(in_intersection),
            int(next_links[0]),
            int(next_links[1]),
            int(next_links[2]),
            target_edge_rel_end_x,
            target_edge_rel_end_y,
            target_edge_norm_len
        ] # 15 componentes
        
        # Añadir información de vehículos cercanos
        for vehicle in nearby_vehicles:
            state_components.extend(vehicle) # 40 componentes

        # Añadir info semáforo
        state_components.extend(next_tls_state) # 3 componentes
        state_components.append(dist_to_tls) # 1 componente

        final_state = np.array(state_components, dtype=np.float32)
        # Verificar dimensión final antes de retornar
        EXPECTED_DIM = 59 # Actualizar según los componentes añadidos
        if final_state.shape[0] != EXPECTED_DIM:
            print(f"ERROR INTERNO: Dimensión de estado inesperada {final_state.shape[0]}, se esperaba {EXPECTED_DIM}")
            # Devolver ceros con la dimensión correcta
            return np.zeros(EXPECTED_DIM, dtype=np.float32)

        return final_state
            
    
    def _get_nearby_vehicles(self, position: Tuple[float, float], radius: float) -> Dict[str, Dict]:
        """Obtiene información de vehículos cercanos en un radio determinado"""
        vehicles = {}
        try:
            all_vehicles = traci.vehicle.getIDList()
            
            for v_id in all_vehicles:
                try:
                    v_pos = traci.vehicle.getPosition(v_id)
                    distance = np.sqrt((position[0] - v_pos[0])**2 + (position[1] - v_pos[1])**2)
                    
                    if distance <= radius:
                        vehicles[v_id] = {
                            'position': v_pos,
                            'speed': traci.vehicle.getSpeed(v_id),
                            'speed_vector': traci.vehicle.getSpeed3D(v_id),
                            'angle': traci.vehicle.getAngle(v_id),
                            'distance': distance
                        }
                except:
                    continue
        except:
            pass
                
        return vehicles
    
    def _calculate_reward(self, prev_position: Optional[Tuple[float, float]], prev_lane: Optional[str], vehicle_exists: bool) -> Tuple[float, Dict]:
        """Calcula la recompensa basada en el estado actual y las acciones tomadas"""
        reward = 0.0
        info = {}
        distance_this_step = 0.0
        dist_to_end = np.inf
        progress_towards_goal = 0.0
        prev_dist_to_end = self.last_dist_to_end
        
        # Penalización por paso
        reward += self.reward_weights["step_reward"]

        if self.forced_collision:
            reward += self.reward_weights["collision"]
            self.collision_count += 1
            info["terminal"] = "forced_collision"
            self.last_dist_to_end = np.inf
            return reward, info
        
        # Si el vehículo no existe, aplicar penalización
        if not vehicle_exists:
            self.last_dist_to_end = np.inf
            # Comprobar si ha sido teleportado (indicador de bloqueo)
            try:
                teleports = traci.simulation.getStartingTeleportIDList()
                if self.vehicle_id in teleports:
                    reward += self.reward_weights["teleport"]
                    self.teleport_count += 1
                    info["terminal"] = "teleported"
                else:
                    # Asumir colisión u otro problema
                    reward += self.reward_weights["collision"]
                    self.collision_count += 1
                    info["terminal"] = "collision"
            except:
                reward += self.reward_weights["collision"]
                info["terminal"] = "unknown_failure"
            
            return reward, info
        
        try:
            # Recompensa por velocidad
            current_position = traci.vehicle.getPosition(self.vehicle_id)
            current_speed = traci.vehicle.getSpeed(self.vehicle_id)
            current_edge = traci.vehicle.getRoadID(self.vehicle_id)
            current_lane = traci.vehicle.getLaneID(self.vehicle_id)
            reward += current_speed * self.reward_weights["speed_reward"]
            is_stopped = current_speed < 0.1
            
            # Recompensa por distancia recorrida
            if prev_position is not None:
                distance_this_step = np.sqrt((current_position[0] - prev_position[0])**2 + 
                                  (current_position[1] - prev_position[1])**2)
                self.total_distance += distance_this_step
                
                # Detectar si va en dirección contraria
                if not is_stopped:
                    route_angle = traci.vehicle.getAngle(self.vehicle_id)
                    movement_vector = (current_position[0] - prev_position[0], 
                                     current_position[1] - prev_position[1])
                    if distance_this_step > 0.1:  # Solo considerar si hay movimiento significativo
                        movement_angle = np.degrees(np.arctan2(movement_vector[1], movement_vector[0]))
                        angle_diff = abs((route_angle - movement_angle + 180) % 360 - 180)
                        if angle_diff > 90:
                            reward += self.reward_weights["wrong_direction"]
                            info["wrong_direction"] = True
            
            # Recompensa por mantener el carril (evitar zigzag innecesario)
            if prev_lane is not None and prev_lane == current_lane and ":" not in current_lane:
                reward += self.reward_weights["keep_lane"]

            # Recompensa por acercarse al final del edge
            if ":" not in current_lane and current_edge != "" and distance_this_step > 0.1: # Solo si se mueve
                try:
                    lane_len = traci.lane.getLength(current_lane)
                    pos_on_lane = traci.vehicle.getLanePosition(self.vehicle_id)
                    dist_to_end = lane_len - pos_on_lane

                    # Recompensa inversamente proporcional a la distancia restante (más recompensa cuanto más cerca)
                    # Añadir un pequeño epsilon para evitar división por cero
                    # reward += (1.0 / (dist_to_end + 1e-2)) * self.reward_weights["approaching_goal"]

                    # Calcular progreso solo si teníamos una distancia válida anterior
                    if prev_dist_to_end is not None and prev_dist_to_end != np.inf:
                        progress_towards_goal = prev_dist_to_end - dist_to_end
                        reward += progress_towards_goal * self.reward_weights["approaching_goal"]
            
                except Exception as e:
                    dist_to_end = np.inf
                    print(f"Error recompensa acercarse al objetivo: {e}")
            # else:
            #     dist_to_end = np.inf

            self.last_dist_to_end = dist_to_end


            # Lógica de recompensa/penalización por semáforo
            is_red_or_amber = False
            tls_info = traci.vehicle.getNextTLS(self.vehicle_id)
            if tls_info:
                tls_id, tls_index, tls_distance, tls_state_char = tls_info[0]
                tls_state = tls_state_char.lower()

                is_close_to_tls = tls_distance < 5.0 # Umbral de cercanía (ajustar)
                is_red_or_amber = tls_state in ['r', 'y']
                if is_close_to_tls and is_red_or_amber:
                    # Recompensa por detenerse en rojo o ambar
                    if is_stopped:
                        reward += self.reward_weights["stopped_at_red"]
                        info["stopped_at_red"] = True
                    
                    # Penalizar pasar en rojo o ámbar a punto de cambiar
                    else:
                        reward += self.reward_weights["ran_red_light"]
                        info["ran_red_light"] = True
                        print(f"¡Penalización! Pasando semáforo {tls_id} en {tls_state} a {tls_distance:.1f}m")

                # Recompensa por pasar en verde
                elif is_close_to_tls and tls_state == 'g' and not is_stopped:
                    reward += self.reward_weights["passed_green_light"]

            # Comprobar paradas de emergencia
            if is_stopped and not is_red_or_amber:
                in_intersection = ":" in current_lane
                if not in_intersection:
                    reward += self.reward_weights["emergency_stop"]
                    self.emergency_stops += 1
                    info["emergency_stop"] = True
                else:
                    reward += self.reward_weights["intersection_stop"]

            
            # Actualizar estadísticas del vehículo
            self.vehicle_stats = {
                "speed": current_speed,
                "position": traci.vehicle.getPosition(self.vehicle_id),
                "angle": traci.vehicle.getAngle(self.vehicle_id),
                "lane": current_lane,
                "edge": traci.vehicle.getRoadID(self.vehicle_id)
            }
            
            # Comprobar si ha llegado al final de la ruta
            try:
                route_index = traci.vehicle.getRouteIndex(self.vehicle_id)
                route = traci.vehicle.getRoute(self.vehicle_id)
                is_last_edge = route and route_index == len(route) - 1

                # Verificar si es el último edge Y si está cerca del final
                if is_last_edge and dist_to_end < 2.0: # Umbral estricto
                    info["segment_goal_reached"] = True
                    reward += self.reward_weights["goal_reached"]
                    print(f"¡Meta del tramo alcanzada! Edge: {current_edge}, Recompensa: +{self.reward_weights['goal_reached']}")

                    # Intentar asignar nueva ruta
                    new_route = self._assign_new_route()
                    print(f"------ new_route: {new_route} ------")

                    if isinstance(new_route, dict) and new_route.get("success") == True:
                        info["terminal"] = "evaluation_route_completed"

                    if new_route is not None:
                        info["new_route_assigned"] = True
                        self.last_dist_to_end = None
                    else:
                        info["terminal"] = "goal_reached_no_new_route"
                        print("Error: no se ha podido asignar una nueva ruta.")
            except Exception as e:
                print(f"Error: {e}")

            if "segment_goal_reached" not in info:
                info["segment_goal_reached"] = False
            
        except traci.exceptions.TraCIException as e_traci:
            # Si el vehículo ya no existe al intentar obtener datos, aplicar penalización grave
            print(f"Error TraCI en _calculate_reward (vehículo probablemente desaparecido): {e_traci}")
            reward += self.reward_weights["collision"]
            info["terminal"] = "vehicle_disappeared_reward"
            self.vehicle_active = False
            self.last_dist_to_end = np.inf

        except Exception as e:
            print(f"Error inesperado al calcular recompensa: {e}")
            self.last_dist_to_end = np.inf
        
        if "segment_goal_reached" not in info:
            info["segment_goal_reached"] = False

        return reward, info
    
    def close(self):
        """Cierra la simulación"""
        port_info = f" (Puerto: {self.port})" if self.port else ""
        if self.simulation_running:
            try:
                if traci.isLoaded():
                    if self.alg_name:
                        self._generate_video()
                    traci.close()
            except:
                print(f"Error durante traci.close(){port_info}: {e}")
            finally:
                self.simulation_running = False
            
    def render(self, mode='human'):
        """
        No es necesario implementar render ya que SUMO-GUI ya proporciona visualización.
        Esta función se mantiene por compatibilidad con entornos de RL.
        """
        pass

    def _save_frame(self):
        """Guarda una captura de pantalla de la simulación en el paso actual."""
        os.makedirs("frames", exist_ok=True)
        frame_filename = f"frames/frame_{self.alg_name}_{self.current_step:04d}.png"
        traci.gui.screenshot("View #0", frame_filename)

    def _generate_video(self):
        comando = [
            "ffmpeg",
            "-framerate", "10",                 # Velocidad de fotogramas
            "-i", "frames/frame_%04d.png",       # Patrón de los archivos de imagen
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-y",
            f"{self.alg_name}_episode.mp4"                        # Nombre del video de salida
        ]
        try:
            subprocess.run(comando, check=True)
            print("Video generado exitosamente: episode.mp4")
        except subprocess.CalledProcessError as e:
            print(f"Error al generar el video: {e}")

    
    @staticmethod
    def generate_eval_routes(output_file: str, num_routes: int = 20, route_length: int = 5):
        print("Generando rutas de evaluación...")
        
        temp_config_path = os.path.abspath(CONFIG['env']['sumo_config_eval'])
        if not os.path.exists(temp_config_path):
            print(f"Error: Archivo de configuración SUMO no encontrado en {temp_config_path}")
            return

        all_generated_routes = []
        temp_env = None
        MAX_NEXT_EDGE_ATTEMPTS = 100

        try:
            sumo_binary = "sumo"
            if os.environ.get("SUMO_HOME"):
                sumo_binary = os.path.join(os.environ["SUMO_HOME"], "bin", sumo_binary)

            # Iniciar SUMO una vez para obtener la lista completa de edges
            print("Obteniendo lista de edges del mapa...")
            traci.start([sumo_binary, "-c", temp_config_path, "--quit-on-end", "--no-step-log"], port=None, label="route_gen_edgelist")
            initial_edges_list = traci.edge.getIDList()
            print(f"Obtenidos {len(initial_edges_list)} edges del mapa.")
            traci.close(False)
            time.sleep(0.5)

            # Crear una instancia de SUMOEnvironment para usar sus métodos de búsqueda
            temp_env = SUMOEnvironment(sumo_config=temp_config_path, gui=False, STATE_DIM=1)

            routes = []
            while len(routes) < num_routes:
                current_route = []

                # Encontrar el primer edge (start_edge)
                start_edge = temp_env._find_random_start_edge(initial_edges_list)
                if not start_edge:
                    print("No se pudo encontrar un start_edge válido, reintentando...")
                    time.sleep(0.1)
                    continue

                current_route.append(start_edge)
                last_edge_in_route = start_edge

                # Encontrar los edges siguientes
                route_valid = True
                for edge_num in range(1, route_length): # Necesitamos route_length - 1 edges más
                    next_edge_found_for_segment = False
                    for attempt in range(MAX_NEXT_EDGE_ATTEMPTS):
                        # Los métodos _find* dentro de temp_env deberían usar la conexión TraCI de temp_env
                        next_edge = temp_env._find_random_next_edge(last_edge_in_route, initial_edges_list)
                        if next_edge and next_edge not in current_route: # Evitar bucles simples inmediatos
                            current_route.append(next_edge)
                            last_edge_in_route = next_edge
                            next_edge_found_for_segment = True
                            break # Salir del bucle de intentos para este segmento
                        else:
                            print(f"  Intento {attempt+1} para edge {edge_num+1} desde {last_edge_in_route} no válido o repetido ({next_edge}).")

                    if not next_edge_found_for_segment:
                        print(f"No se pudo encontrar el edge {edge_num+1} para la ruta actual comenzando con {current_route[0]}. Descartando ruta.")
                        route_valid = False
                        break # Salir del bucle de construcción de esta ruta

                if route_valid and len(current_route) == route_length:
                    all_generated_routes.append(current_route) # Guardar como lista de strings
                    routes.append(current_route)
                    print(f"Ruta {len(routes)}/{num_routes} encontrada: {current_route}")
                else:
                    print(f"Ruta descartada (longitud: {len(current_route)}): {current_route}")


            if len(all_generated_routes) < num_routes:
                print(f"ADVERTENCIA: Solo se generaron {len(all_generated_routes)} de las {num_routes} rutas largas solicitadas.")

            temp_env.close() # Cerrar el entorno temporal
            print("\nLista de rutas generada:")
            print("EVAL_ROUTES = [")
            for r in routes:
                print(f"    ('{r[0]}', '{r[1]}'),")
            print("]")

            with open(output_file, 'w') as f:
                json.dump(routes, f, indent=4)

        except Exception as e:
            print(f"Error generando rutas: {e}")
            if traci.isLoaded():
                traci.close()
            raise Exception(e)

    @staticmethod
    def load_eval_routes_long(routes_filepath: str):
        with open(routes_filepath, 'r') as f:
            return json.load(f)

