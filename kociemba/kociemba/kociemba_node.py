import importlib.util
import site
import sys
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


def load_external_kociemba():
    current_package_dir = Path(__file__).resolve().parent
    candidate_site_packages = []

    try:
        candidate_site_packages.append(Path(site.getusersitepackages()))
    except AttributeError:
        pass

    for site_path in getattr(site, 'getsitepackages', lambda: [])():
        candidate_site_packages.append(Path(site_path))

    for path_entry in sys.path:
        if 'site-packages' in path_entry:
            candidate_site_packages.append(Path(path_entry))

    seen_paths = set()
    for site_packages_dir in candidate_site_packages:
        resolved_dir = site_packages_dir.resolve()
        if resolved_dir in seen_paths:
            continue
        seen_paths.add(resolved_dir)

        init_file = resolved_dir / 'kociemba' / '__init__.py'
        if not init_file.is_file():
            continue
        if init_file.resolve().parent == current_package_dir:
            continue

        spec = importlib.util.spec_from_file_location(
            'kociemba_external',
            init_file,
            submodule_search_locations=[str(init_file.parent)],
        )
        if spec is None or spec.loader is None:
            continue

        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        if hasattr(module, 'solve'):
            return module

    raise ImportError(
        'Unable to locate the external kociemba solver package. '
        'Install it with "python3 -m pip install kociemba".'
    )


class KociembaNode(Node):
    def __init__(self):
        super().__init__('kociemba_node')
        self.kociemba_solver = load_external_kociemba()
        self.subscription = self.create_subscription(
            String,
            'cube_result',
            self.cube_state_callback,
            10)
        self.publisher_ = self.create_publisher(String, 'cube_solution', 10)
        self.get_logger().info(
            f'KociembaNode has been started. Solver loaded from {self.kociemba_solver.__file__}'
        )

    def cube_state_callback(self, msg):
        cube_state = msg.data.strip()
        if not cube_state:
            self.get_logger().warn('Received an empty cube_result message.')
            return
        if len(cube_state) != 54:
            self.get_logger().warn(
                f'Received cube string with length {len(cube_state)} instead of 54: {cube_state}'
            )
            return
        try:
            solution = self.kociemba_solver.solve(cube_state)
        except Exception as exc:
            self.get_logger().error(f'Failed to solve cube state "{cube_state}": {exc}')
            return

        solution_msg = String()
        solution_msg.data = solution
        self.publisher_.publish(solution_msg)
        self.get_logger().info(f'Published cube solution: {solution}')


def main(args=None):
    rclpy.init(args=args)
    kociemba_node = KociembaNode()
    rclpy.spin(kociemba_node)
    kociemba_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
