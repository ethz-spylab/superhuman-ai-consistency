from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import chess
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

# from chess.engine import AnalysisResult
if TYPE_CHECKING:
    from rl_testing.engine_generators.relaxed_uci_protocol import ExtendedAnalysisResult


class TreeParser:
    # Every line containing node/tree info should begin with this
    # string.
    PARSE_TOKEN = "TREE INFO"
    START_TREE_TOKEN = "START TREE"
    END_TREE_TOKEN = "END TREE"
    START_NODE_TOKEN = "START NODE:"
    END_NODE_TOKEN = "END NODE"
    NODE_TOKEN = "node"
    POSITION_TOKEN = "POSITION:"

    attribute_map = {
        "N": "num_visits",
        "move": "move",
        "IN_FLIGHT": "in_flight_visits",
        "P": "policy_value",
        "WL": "w_minus_l",
        "D": "draw_value",
        "M": "num_moves_left",
        "Q": "q_value",
        "U": "u_value",
        "S": "s_value",
        "V": "v_value",
        "T": "is_terminal",
    }

    def __init__(self, analysis_result: "ExtendedAnalysisResult") -> None:
        self.tree: Optional[TreeInfo] = None
        self.node: Optional[NodeInfo] = None
        self.parent_id: Optional[int] = None
        self.node_cache: dict[int, NodeInfo] = {}
        self.analysis_result: "ExtendedAnalysisResult" = analysis_result

    def is_line_parsable(self, line: str) -> bool:
        return self.PARSE_TOKEN in line

    def parse_line(self, line: str) -> None:
        # Remove the PARSE_TOKEN and everything before it from the line.
        line = line.split(self.PARSE_TOKEN, 1)[1]

        if self.START_TREE_TOKEN in line:
            self.start_tree()
        elif self.END_TREE_TOKEN in line:
            self.end_tree()
        elif self.START_NODE_TOKEN in line:
            self.start_node(line)
        elif self.END_NODE_TOKEN in line:
            self.end_node()
        elif self.POSITION_TOKEN in line:
            self.parse_position(line)
        elif self.NODE_TOKEN in line:
            self.parse_node_line(line)
        else:
            self.parse_edge_line(line)

    def start_tree(self) -> None:
        self.tree = TreeInfo()

    def end_tree(self) -> None:
        assert self.tree is not None
        assert self.tree.root_node is not None

        # Compute the depth of each node
        num_per_depth = []
        self.tree.root_node.assign_depth(0, num_per_depth)

        # Store the node cache
        self.tree.node_cache = self.node_cache

        # Add the tree to the analysis result
        self.analysis_result.mcts_tree = self.tree
        for index in range(len(self.analysis_result.multipv)):
            self.analysis_result.multipv[index]["mcts_tree"] = self.tree

        # Print some stats
        # print(f"Node counter: {self.node_counter}")
        # print(f"Node duplicate counter: {self.node_duplicate_counter}")
        # print(f"Difference: {self.node_counter - self.node_duplicate_counter}")

        # Reset the parser
        self.tree = None
        self.node_cache: Dict[int, NodeInfo] = {}

    def start_node(self, line: str) -> None:
        # Remove the INDEX_TOKEN and everything before it from the line.
        line = line.split(self.START_NODE_TOKEN, 1)[1]

        # Split the line into the index part, the parent part and the multi-visit part
        visit_index, remainder = line.split("PARENT:")
        parent, multi_visit = remainder.split("VISITED_BEFORE:")
        visit_index, parent, multi_visit = (
            int(visit_index.strip()),
            int(parent.strip()),
            int(multi_visit.strip()),
        )

        assert visit_index > parent, "Parent index is not smaller than child index!"

        # If the multi-visit part is not 0, then log a warning
        if multi_visit != 0:
            print(f"WARNING: Multi-visit is not 0: {multi_visit}")

        # Create the node
        self.node = NodeInfo(visit_index)
        self.parent_id = parent

        # Set the node as the root node if its index is 0
        if visit_index == 0:
            self.tree.root_node = self.node

        # Add the node to the node cache
        assert visit_index not in self.node_cache, "Node index already in cache!"
        self.node_cache[visit_index] = self.node

    def end_node(self) -> None:
        assert self.node.fen is not None, "FEN is not set!"
        self.node = None
        self.parent_id = None

    def parse_position(self, line: str) -> None:
        # Remove the POSITION_TOKEN and everything before it from the line.
        line = line.split(self.POSITION_TOKEN, 1)[1]

        # Parse the line if the line contains a fen string.
        if "/" in line:
            self.node.set_fen(line.strip())

        # Parse the line if the line contains a list of moves.
        elif "+" in line:
            root_board = chess.Board(self.tree.root_node.fen)
            move_list = line.split("+")
            move_list = [move.strip() for move in move_list]
            move_list = move_list[1:]
            for move in move_list:
                root_board.push_uci(move)

            self.node.set_fen(root_board.fen(en_passant="fen"))

        # Now that the position is set, we can connect the node to its parent
        # Get the parent node from the cache
        if self.parent_id != -1:
            parent_node = self.node_cache[self.parent_id]
            parent_node.connect_child_node(self.node)

    def parse_data_line(self, line: str) -> None:
        # Remove the node or edge token
        line = line.strip()
        line = line[line.index("(") + 1 : -1]
        if line[-1] == "T":
            line = line[:-4]
        tokens = line.split(") (")

        result_dict = {}
        for token in tokens:
            key, value = token.split(":")
            key, value = key.strip(), value.strip()

            if value.endswith("%"):
                value = float(value[:-1]) / 100
            elif value.startswith("+"):
                value = int(value[1:])
            elif "-.-" in value:
                value = None
            elif key == "move":
                if value != "node":
                    value = chess.Move.from_uci(value)
                else:
                    value = None
            else:
                try:
                    value = float(value)
                except ValueError:
                    raise ValueError(f"Can't parse value {value}")

            if value is not None:
                result_dict[self.attribute_map[key]] = value

        return result_dict

    def parse_node_line(self, line: str) -> None:
        attribute_dict = self.parse_data_line(line)

        for attribute in attribute_dict:
            setattr(self.node, attribute, attribute_dict[attribute])

        self.node.check_required_attributes()

    def parse_edge_line(self, line: str) -> None:
        attribute_dict = self.parse_data_line(line)

        # Create a new edge
        edge = EdgeInfo(attribute_dict["move"], self.node)
        del attribute_dict["move"]

        for attribute in attribute_dict:
            setattr(edge, attribute, attribute_dict[attribute])

        edge.check_required_attributes()


class OneNodeParser(TreeParser):
    EDGE_TOKEN = "S:"  # A substring that is present in every edge line
    NODE_TOKEN = "node"  # A substring that is present in every node line

    def __init__(self, analysis_result: "ExtendedAnalysisResult") -> None:
        self.node: NodeInfo = NodeInfo(visit_index=0)
        self.analysis_result: "ExtendedAnalysisResult" = analysis_result

    def is_line_parsable(self, line: str) -> bool:
        return "string" in line and (self.EDGE_TOKEN in line or self.NODE_TOKEN in line)

    def parse_line(self, line: str) -> None:
        # TODO: CHANGE THIS!
        if self.EDGE_TOKEN in line:
            self.parse_edge_line(line)
        elif self.NODE_TOKEN in line:
            self.parse_node_line(line)
        else:
            raise ValueError(f"Can't parse line: {line}")

    def start_tree(self) -> None:
        raise NotImplementedError("This class does not support parsing trees.")

    def end_tree(self) -> None:
        raise NotImplementedError("This class does not support parsing trees.")

    def start_node(self) -> None:
        raise NotImplementedError("This class does not require a start_node function.")

    def end_node(self) -> None:
        # Compute the depth of each node
        num_per_depth = []
        self.node.assign_depth(0, num_per_depth)

        # Add the node to the analysis result
        self.analysis_result.root_and_child_scores = self.node
        for index in range(len(self.analysis_result.multipv)):
            self.analysis_result.multipv[index]["root_and_child_scores"] = self.node

        # Reset the parser
        self.node = NodeInfo(visit_index=0)

    def parse_position(self, line: str) -> None:
        raise NotImplementedError("This class does not require a parse_position function.")

    def preprocess_line(self, line: str) -> str:
        # Remove the "info string" prefix from the line
        line = line.split("info string ", 1)[1]

        # Add a name to the first token
        line = "move:" + line

        # If the last token is a terminal token, add a value to it
        if line[-1] == "T":
            line = line[:-4] + ":True"
        else:
            line = line + " T:False"

        # Convert values of type "+0" to "IN_FLIGHT:0"
        line = line.replace("+", "IN_FLIGHT:")

        bracket_open = False
        fixed_line = []
        # First make sure that all the data is in the form "(...)"
        for index, char in enumerate(line):
            if not str.isspace(char) and char != "(" and not bracket_open:
                fixed_line.append("(")
                bracket_open = True
            elif not str.isspace(char) and char == "(" and bracket_open:
                fixed_line.append(")")
            if char == ")":
                bracket_open = False
            elif char == "(":
                bracket_open = True

            fixed_line.append(char)

        if bracket_open:
            fixed_line.append(")")

        # Remove all white spaces
        fixed_line = "".join(fixed_line)
        fixed_line = fixed_line.replace(" ", "")

        return fixed_line

    def parse_data_line(self, line: str) -> None:
        # TODO: FIX THIS!
        line = line.strip()
        line = line[line.index("(") + 1 : -1]
        tokens = line.split(")(")
        tokens = [token for token in tokens if ":" in token]

        result_dict = {}
        for token in tokens:
            key, value = token.split(":")
            key, value = key.strip(), value.strip()

            if value.endswith("%"):
                value = float(value[:-1]) / 100
            elif value.startswith("+"):
                value = int(value[1:])
            elif "-.-" in value:
                value = None
            elif key == "T":
                value = value == "True"
            elif key == "move":
                if value != "node":
                    value = chess.Move.from_uci(value)
                else:
                    value = None
            else:
                try:
                    value = float(value)
                except ValueError:
                    raise ValueError(f"Can't parse value {value}")

            if value is not None:
                result_dict[self.attribute_map[key]] = value

        return result_dict

    def parse_node_line(self, line: str) -> None:
        line = self.preprocess_line(line)
        attribute_dict = self.parse_data_line(line)

        for attribute in attribute_dict:
            setattr(self.node, attribute, attribute_dict[attribute])

        self.node.check_required_attributes()
        self.end_node()

    def parse_edge_line(self, line: str) -> None:
        line = self.preprocess_line(line)
        attribute_dict = self.parse_data_line(line)

        # Create a new edge
        edge = EdgeInfo(attribute_dict["move"], self.node)
        del attribute_dict["move"]

        for attribute in attribute_dict:
            setattr(edge, attribute, attribute_dict[attribute])

        edge.check_required_attributes()


class Info:
    required_attributes = []

    def __init__(self) -> None:
        # Initialize the node data
        self.num_visits: Optional[int] = None
        self.in_flight_visits: Optional[int] = None
        self.policy_value: Optional[float] = None
        self.w_minus_l: Optional[float] = None
        self.draw_value: Optional[float] = None
        self.num_moves_left: Optional[int] = None
        self.q_value: Optional[float] = None
        self.u_value: Optional[float] = None
        self.s_value: Optional[float] = None
        self.v_value: Optional[float] = None

    def check_required_attributes(self):
        for attribute_name in self.required_attributes:
            if getattr(self, attribute_name) is None:
                raise AttributeError(f"Attribute {attribute_name} must be set")


class TreeInfo:
    def __init__(self) -> None:
        self.root_node: Optional[NodeInfo] = None
        self.node_cache: dict[str, NodeInfo] = {}


class NodeInfo(Info):

    required_attributes = [
        "num_visits",
        "in_flight_visits",
        "policy_value",
        "w_minus_l",
        "draw_value",
        "num_moves_left",
        "q_value",
        "v_value",
        "is_also_terminal",
        "contains_only_terminal",
    ]

    def __init__(self, visit_index: int) -> None:
        super().__init__()
        # Initialize parent and child edges
        self.parent_edge: Optional[EdgeInfo] = None
        self.child_edges: List[EdgeInfo] = []

        # Initialize the board position
        self.fen: Optional[str] = None
        self.visit_index = visit_index

        # Initialize depth information
        self.depth = -1
        self.depth_index = -1

        # This value indicates whether this node can also be a terminal node.
        self.is_also_terminal = False
        self.contains_only_terminal = False

    @property
    def child_nodes(self) -> List["NodeInfo"]:
        return [edge.end_node for edge in self.child_edges if edge.end_node is not None]

    @property
    def parent_node(self) -> "NodeInfo":
        assert self.parent_edge is not None
        return self.parent_edge.start_node

    @property
    def orphan_edges(self) -> List["NodeInfo"]:
        return [edge for edge in self.child_edges if edge.end_node is None]

    def set_fen(self, fen: str) -> None:
        # Check if the fen string is valid.
        temp_board = chess.Board(fen)
        if temp_board.is_valid():
            self.fen = fen
        else:
            raise ValueError(f"Fen string {fen} is not valid.")

    def connect_child_node(self, child_node: "NodeInfo") -> None:
        board = chess.Board(self.fen)

        # Find the edge that connects the child node
        for edge in self.child_edges:
            board.push(edge.move)
            if board.fen(en_passant="fen") == child_node.fen:
                if edge.end_node is None:
                    edge.set_end_node(child_node)
                return
            board.pop()

        raise ValueError(f"Can't find edge to connect {child_node.fen} to {self.fen}.")

    def assign_depth(self, depth: int, num_per_depth: List[int]):
        # Assign your own depth
        self.depth = depth

        # Compute how many other nodes already have this depth
        if len(num_per_depth) <= depth:
            num_per_depth.append(0)
        self.depth_index = num_per_depth[depth]
        num_per_depth[depth] += 1

        # Assign the depths of the child nodes
        for edge in self.child_edges:
            if edge.end_node is not None:
                edge.end_node.assign_depth(max(depth + 1, edge.end_node.depth), num_per_depth)


class EdgeInfo(Info):
    required_attributes = [
        "start_node",
        "num_visits",
        "in_flight_visits",
        "policy_value",
        "q_value",
        "u_value",
        "s_value",
    ]

    def __init__(
        self,
        move: Union[str, chess.Move],
        start_node: Optional[NodeInfo] = None,
        end_node: Optional[NodeInfo] = None,
    ) -> None:
        super().__init__()
        # Initialize the start and end nodes
        self.start_node: Optional[NodeInfo] = None
        self.end_node: Optional[NodeInfo] = None

        if start_node is not None:
            self.set_start_node(start_node)
        if end_node is not None:
            self.set_end_node(end_node)

        # Initialize the move
        if isinstance(move, str):
            self.move: chess.Move = chess.Move.from_uci(move)
        elif isinstance(move, chess.Move):
            self.move = move
        else:
            raise TypeError(f"Move must be a string or chess.Move, not {type(move)}")

    def set_start_node(self, node: NodeInfo) -> None:
        self.start_node = node
        node.child_edges.append(self)

    def set_end_node(self, node: NodeInfo) -> None:
        assert node.parent_edge is None
        self.end_node = node
        node.parent_edge = self


def convert_tree_to_networkx(tree: TreeInfo, only_basic_info: bool = False) -> nx.DiGraph:
    red = np.array([255, 0, 0])
    green = np.array([0, 255, 0])
    white = np.array([255, 255, 255])

    graph = nx.Graph()
    # Add all nodes to the graph
    for index in tree.node_cache:
        node = tree.node_cache[index]

        if node.v_value is None:
            print("This should not happen!")

        # Compute the color of the new node
        node_value_current_player = node.v_value if node.depth % 2 == 0 else -node.v_value
        if node_value_current_player <= 0:
            color = red + (white - red) * (1 + node_value_current_player)
        else:
            color = green + (white - green) * (1 - node_value_current_player)
        color = color.round().astype(int)
        color_str = f"#{color[0]:0{2}x}{color[1]:0{2}x}{color[2]:0{2}x}"

        x, y = node.depth_index * 10, node.depth * 5

        graph.add_node(
            node.visit_index if only_basic_info else node,
            color=color_str,
            x=x,
            y=y,
            pos=(x, y),
        )
    for index in tree.node_cache:
        node = tree.node_cache[index]
        for edge in node.child_edges:
            if edge.end_node is not None:
                if only_basic_info:
                    assert edge.start_node.visit_index in graph
                    assert edge.end_node.visit_index in graph
                    graph.add_edge(
                        edge.start_node.visit_index,
                        edge.end_node.visit_index,
                        size=edge.q_value,
                    )
                else:
                    assert edge.start_node in graph
                    assert edge.end_node in graph
                    graph.add_edge(edge.start_node, edge.end_node, size=edge.q_value)

    return graph


def plot_networkx_tree(tree: TreeInfo, only_basic_info: bool = False) -> None:
    graph = convert_tree_to_networkx(tree, only_basic_info)
    pos = nx.get_node_attributes(graph, "pos")

    # Flip the y axis
    for key in pos:
        pos[key] = (pos[key][0], -pos[key][1])
    colors = nx.get_node_attributes(graph, "color").values()
    sizes = 1  # [graph[u][v]["size"] * 5 for u, v in graph.edges]
    nx.draw(
        graph,
        pos,
        node_color=colors,
        # Make the node border black
        edgecolors="black",
        with_labels=False,
        node_size=100,
        width=sizes,
        arrowsize=10,
    )

    plt.show()
