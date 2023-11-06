mod data;

use std::cmp::{Ordering, Reverse};
use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use std::time::Instant;
use itertools::{Itertools, izip};
use priority_queue::PriorityQueue;

#[derive(Debug)]
#[derive(Clone)]
#[derive(Hash)]
#[derive(Eq)]
#[derive(PartialEq)]
#[derive(Ord)]
#[derive(PartialOrd)]
struct BatteryState {
    level: u32,
}

#[derive(Debug)]
#[derive(Clone)]
#[derive(Hash)]
#[derive(Eq)]
#[derive(PartialEq)]
#[derive(Ord)]
#[derive(PartialOrd)]
struct ApplianceState {
    active_cycle: u32,
    completed_cycles: u32,
}

#[derive(Debug)]
#[derive(Clone)]
#[derive(Hash)]
#[derive(Eq)]
#[derive(PartialEq)]
#[derive(Ord)]
#[derive(PartialOrd)]
struct HomeState {
    timestep: u32,
    battery: BatteryState,
    appliances: Vec<ApplianceState>,
}

#[derive(Debug)]
#[derive(Clone)]
#[derive(PartialEq)]
enum BatteryAction {
    DISCHARGE,
    OFF,
    CHARGE,
}

#[derive(Debug)]
#[derive(Clone)]
#[derive(PartialEq)]
enum ApplianceAction {
    OFF,
    ON,
}

#[derive(Debug)]
#[derive(Clone)]
struct HomeAction {
    battery: BatteryAction,
    appliances: Vec<ApplianceAction>,
}

struct BatteryParameters {
    capacity: u32,
    rate: f64,
    initial_level: u32,
    min_required_level: u32,
}

impl BatteryParameters {
    fn new(capacity: u32, rate: f64, initial_level: u32, min_required_level: u32) -> Self {
        assert!(rate > 0.0);
        assert!(initial_level < capacity);
        assert!(min_required_level < capacity);
        return Self { capacity, rate, initial_level, min_required_level };
    }
}

struct ApplianceParameters {
    label: String,
    duration: u32,
    rate: f64,
    min_required_cycles: u32,
}

impl ApplianceParameters {
    fn new(label: String, duration: u32, rate: f64, min_required_cycles: u32) -> Self {
        assert!(duration > 0);
        assert!(rate > 0.0);
        return Self { label, duration, rate, min_required_cycles };
    }
}

struct HomeParameters {
    horizon: u32,
    battery: BatteryParameters,
    appliances: Vec<ApplianceParameters>,
}

trait PlanningProblem<State, Action> {
    fn initial_state(&self) -> State;
    fn applicable_actions(&self, state: &State) -> Vec<&Action>;
    fn transition_function(&self, state: &State, action: &Action) -> State;
    fn cost_function(&self, state: &State, action: &Action, successor_state: &State) -> f64;
    fn is_goal(&self, state: &State) -> bool;
}

struct HomeProblem {
    home_parameters: HomeParameters,
    import_prices: Vec<f64>,
    export_prices: Vec<f64>,
    min_real_cost: Vec<f64>,
    min_required_timesteps: Vec<u32>,
    available_actions: Vec<HomeAction>,
}

impl HomeProblem {
    fn new(home_parameters: HomeParameters) -> Self {
        let horizon = usize::try_from(home_parameters.horizon).unwrap();
        let import_prices = data::IMPORT_PRICES[0..horizon].to_vec();
        let export_prices = data::EXPORT_PRICES[0..horizon].to_vec();

        let mut max_import_energy = 0.0;
        max_import_energy += &home_parameters.battery.rate;
        for appliance_parameters in &home_parameters.appliances {
            max_import_energy += appliance_parameters.rate;
        }

        let max_export_energy = -home_parameters.battery.rate;

        let mut min_real_cost = vec![];
        for (import_price, export_price) in izip!(&import_prices, &export_prices) {
            let option1 = max_import_energy * import_price;
            let option2 = max_export_energy * export_price;
            min_real_cost.push(if option1 <= option2 { option1 } else { option2 });
        }

        let mut min_required_timesteps = vec![];
        for appliance_parameters in &home_parameters.appliances {
            min_required_timesteps.push(appliance_parameters.min_required_cycles * appliance_parameters.duration);
        }

        let battery_actions = vec![BatteryAction::DISCHARGE, BatteryAction::OFF, BatteryAction::CHARGE];
        let appliance_actions = vec![ApplianceAction::OFF, ApplianceAction::ON];
        let mut available_actions = vec![];
        if home_parameters.appliances.is_empty() {
            for battery_action in battery_actions {
                available_actions.push(HomeAction { battery: battery_action.clone(), appliances: vec![] });
            }
        } else {
            for battery_action in battery_actions {
                for appliance_action_tuple in (0..home_parameters.appliances.len()).map(|_| &appliance_actions).multi_cartesian_product() {
                    available_actions.push(HomeAction { battery: battery_action.clone(), appliances: appliance_action_tuple.into_iter().cloned().collect() });
                }
            }
        }

        return Self { home_parameters, import_prices, export_prices, min_real_cost, min_required_timesteps, available_actions }
    }

    fn is_applicable(&self, state: &HomeState, action: &HomeAction) -> bool {
        if state.battery.level <= 0 && action.battery == BatteryAction::DISCHARGE {
            return false;
        }
        if state.battery.level >= self.home_parameters.battery.capacity && action.battery == BatteryAction::CHARGE {
            return false;
        }
        for (appliance_parameters, appliance_state, appliance_action) in izip!(&self.home_parameters.appliances, &state.appliances, &action.appliances) {
            if appliance_state.active_cycle > 0 && *appliance_action == ApplianceAction::OFF {
                return false;
            }
            if state.timestep + appliance_parameters.duration - appliance_state.active_cycle - 1 >= self.home_parameters.horizon && *appliance_action == ApplianceAction::ON {
                return false;
            }
        }
        return true;
    }

    fn heuristic_function(&self, state: &HomeState) -> f64 {
        let timesteps_to_horizon = self.home_parameters.horizon - state.timestep;

        if state.battery.level + timesteps_to_horizon < self.home_parameters.battery.min_required_level {
            return f64::INFINITY;
        }

        for (appliance_parameters, appliance_state, appliance_min_required_timesteps) in izip!(&self.home_parameters.appliances, &state.appliances, &self.min_required_timesteps) {
            let appliance_timesteps_completed = (appliance_state.completed_cycles * appliance_parameters.duration) + appliance_state.active_cycle;

            if appliance_timesteps_completed > *appliance_min_required_timesteps {
                return f64::INFINITY;
            }

            let appliance_timesteps_to_go = appliance_min_required_timesteps - appliance_timesteps_completed;

            if appliance_timesteps_to_go > timesteps_to_horizon {
                return f64::INFINITY;
            }
        }

        if timesteps_to_horizon == 0 {
            return 0.0;
        }

        let mut future_expense_lower_bound = 0.0;

        let mut appliance_active_cycle_energy = vec![];
        for (appliance_parameters, appliance_state) in izip!(&self.home_parameters.appliances, &state.appliances) {
            let mut x = vec![];
            if appliance_state.active_cycle > 0 {
                for _ in 0..appliance_parameters.duration - appliance_state.active_cycle {
                    x.push(appliance_parameters.rate);
                }
            }
            appliance_active_cycle_energy.push(x);
        }

        let mut active_cycles_energy = vec![];
        let mut lengths = vec![];
        let mut longest = 0;
        for appliance in &appliance_active_cycle_energy {
            let appliance_len = appliance.len() as u32;
            lengths.push(appliance_len);
            if appliance_len > longest {
                longest = appliance_len;
            }
        }
        for i in 0..longest {
            let index = usize::try_from(i).unwrap();
            let mut value = 0.0;
            for (appliance, appliance_length) in  izip!(&appliance_active_cycle_energy, &lengths) {
                if *appliance_length > i {
                    value += appliance[index];
                }
            }
            active_cycles_energy.push(value);
        }

        let timestep = usize::try_from(state.timestep).unwrap();
        for (timestep_offset, (import_price, export_price, off_energy)) in izip!(&self.import_prices[timestep..], &self.export_prices[timestep..], active_cycles_energy).enumerate() {
            let discharge_energy = off_energy - self.home_parameters.battery.rate;
            let charge_energy = off_energy + self.home_parameters.battery.rate;
            let options = vec![
                if discharge_energy >= 0.0 { discharge_energy * import_price } else { discharge_energy * export_price },
                if off_energy >= 0.0 { off_energy * import_price } else { off_energy * export_price },
                if charge_energy >= 0.0 { charge_energy * import_price } else { charge_energy * export_price },
            ];
            let mut min_option = f64::INFINITY;
            for option in options {
                if option < min_option {
                    min_option = option;
                }
            }
            future_expense_lower_bound += min_option - self.min_real_cost[timestep + timestep_offset];
        }

        return future_expense_lower_bound;
    }

    fn real_cost(&self, plan: &Vec<HomeAction>) -> f64 {
        let mut total_real_cost = 0.0;

        for (timestep, action) in plan.iter().enumerate() {
            let mut step_energy = 0.0;

            if action.battery == BatteryAction::CHARGE {
                step_energy += self.home_parameters.battery.rate;
            } else if action.battery == BatteryAction::DISCHARGE {
                step_energy -= self.home_parameters.battery.rate;
            }

            for (appliance_parameters, appliance_action) in izip!(&self.home_parameters.appliances, &action.appliances) {
                if *appliance_action == ApplianceAction::ON {
                    step_energy += appliance_parameters.rate;
                }
            }

            total_real_cost += if step_energy >= 0.0 { step_energy * self.import_prices[timestep] } else { step_energy * self.export_prices[timestep] }
        }

        return total_real_cost;
    }
}

impl PlanningProblem<HomeState, HomeAction> for HomeProblem {
    fn initial_state(&self) -> HomeState {
        let mut appliance_states = vec![];
        for _ in &self.home_parameters.appliances {
            appliance_states.push(ApplianceState { active_cycle: 0, completed_cycles: 0 });
        }
        return HomeState {
            timestep: 0,
            battery: BatteryState { level: self.home_parameters.battery.initial_level },
            appliances: appliance_states,
        };
    }

    fn applicable_actions(&self, state: &HomeState) -> Vec<&HomeAction> {
        let mut applicable_actions = vec![];
        if state.timestep >= self.home_parameters.horizon {
            return applicable_actions;
        }

        for action in &self.available_actions {
            if self.is_applicable(&state, action) {
                applicable_actions.push(action);
            }
        }
        return applicable_actions;
    }

    fn transition_function(&self, state: &HomeState, action: &HomeAction) -> HomeState {
        let successor_battery_state = BatteryState { level: if action.battery == BatteryAction::CHARGE { state.battery.level + 1 } else if action.battery == BatteryAction::DISCHARGE { state.battery.level - 1 } else { state.battery.level } };

        let mut successor_appliance_states = vec![];
        for (appliance_parameters, appliance_state, appliance_action) in izip!(&self.home_parameters.appliances, &state.appliances, &action.appliances) {
            if *appliance_action == ApplianceAction::ON {
                if appliance_state.active_cycle == appliance_parameters.duration - 1 {  // cycle ending (and possibly starting as well)
                    successor_appliance_states.push(ApplianceState { active_cycle: 0, completed_cycles: appliance_state.completed_cycles + 1 });
                } else if appliance_state.active_cycle == 0 {  // cycle starting (but not ending)
                    successor_appliance_states.push(ApplianceState { active_cycle: 1, completed_cycles: appliance_state.completed_cycles });
                } else {
                    successor_appliance_states.push(ApplianceState { active_cycle: appliance_state.active_cycle + 1, completed_cycles: appliance_state.completed_cycles });
                }
            } else {
                successor_appliance_states.push(ApplianceState { active_cycle: 0, completed_cycles: appliance_state.completed_cycles });
            }
        }

        return HomeState {
            timestep: state.timestep + 1,
            battery: successor_battery_state,
            appliances: successor_appliance_states,
        };
    }

    fn cost_function(&self, state: &HomeState, action: &HomeAction, _successor_state: &HomeState) -> f64 {
        let mut energy = 0.0;

        if action.battery == BatteryAction::CHARGE {
            energy += self.home_parameters.battery.rate;
        } else if action.battery == BatteryAction::DISCHARGE {
            energy -= self.home_parameters.battery.rate;
        }

        for (appliance_parameters, appliance_action) in izip!(&self.home_parameters.appliances, &action.appliances) {
            if *appliance_action == ApplianceAction::ON {
                energy += appliance_parameters.rate;
            }
        }

        let timestep = usize::try_from(state.timestep).unwrap();
        let real_cost = if energy >= 0.0 { energy * self.import_prices[timestep] } else { energy * self.export_prices[timestep] };
        return real_cost - self.min_real_cost[timestep];
    }

    fn is_goal(&self, state: &HomeState) -> bool {
        if state.timestep != self.home_parameters.horizon {
            return false;
        }
        if state.battery.level < self.home_parameters.battery.min_required_level {
            return false;
        }
        for (appliance_parameters, appliance_state) in izip!(&self.home_parameters.appliances, &state.appliances) {
            if appliance_state.completed_cycles < appliance_parameters.min_required_cycles {
                return false;
            }
        }
        return true;
    }
}

// https://github.com/garro95/priority-queue/issues/27#issuecomment-743745069
#[derive(PartialOrd, PartialEq)]
pub struct MyF64(f64);

impl Eq for MyF64 {}

impl Ord for MyF64 {
    fn cmp(&self, other: &Self) -> Ordering {
        if let Some(ordering) = self.partial_cmp(other) {
            ordering
        } else {
            // Choose what to do with NaNs, for example:
            Ordering::Less
        }
    }
}

struct Parent<State, Action> {
    state: State,
    action: Action,
}

struct Node<State, Action> {
    path_cost: f64,
    evaluation: f64,
    depth: u32,
    parent: Option<Parent<State, Action>>,
}

fn _reconstruct_solution<State: Hash + Eq, Action: Clone>(nodes: &HashMap<State, Node<State, Action>>, terminal_state: &State) -> Option<(Vec<Action>, f64)> {
    let mut plan_back = vec![];
    let mut state = terminal_state;
    while nodes.contains_key(&state) {
        let optional_parent = &nodes.get(&state).unwrap().parent;
        if optional_parent.is_none() {  // reached initial state
            return Some((plan_back.into_iter().rev().collect(), nodes.get(&terminal_state).unwrap().path_cost));
        }
        let parent = optional_parent.as_ref().unwrap();
        state = &parent.state;
        let action = &parent.action;
        plan_back.push(action.clone());
    }
    return None;
}

fn _best_first_search<State: Hash + Eq + Clone, Action: Clone>(planning_problem: &impl PlanningProblem<State, Action>, evaluation_function: impl Fn(f64, &State) -> f64, verbose: bool) -> Option<(Vec<Action>, f64)> {
    let start_time = Instant::now();

    let initial_state: State = planning_problem.initial_state();

    let mut nodes: HashMap<State, Node<State, Action>> = HashMap::new();
    nodes.insert(initial_state.clone(), Node {
        path_cost: 0.0,
        evaluation: evaluation_function(0.0, &initial_state),
        depth: 0,
        parent: None,
    });

    let mut insertion_index: u32 = 0;

    let mut frontier: PriorityQueue<(u32, State), Reverse<MyF64>> = PriorityQueue::new();
    frontier.push((insertion_index, initial_state.clone()), Reverse(MyF64(nodes.get(&initial_state).unwrap().evaluation)));
    let mut frontier_items = HashSet::new();
    frontier_items.insert(initial_state.clone());
    insertion_index += 1;

    let mut max_depth: u32 = 0;
    let mut max_states_visited: u32 = 0;
    let mut previous_max_depth: u32 = 0;
    let mut previous_max_states_visited: u32 = 0;

    while !frontier.is_empty() {
        let selected_state = frontier.pop().unwrap().0.1;  // tuple = ((insertion_index, state), priority)
        frontier_items.remove(&selected_state);

        if verbose {
            let current_depth = nodes.get(&selected_state).unwrap().depth;
            if current_depth > max_depth {
                max_depth = current_depth;
            }
            let states_visited = nodes.len() as u32;
            if states_visited > max_states_visited {
                max_states_visited = states_visited;
            }
            if max_depth > previous_max_depth || max_states_visited > (previous_max_states_visited as f64 * 1.1) as u32 {
                let elapsed_time = start_time.elapsed();
                println!("max depth: {:?}, states visited: {:?}, elapsed time: {:?}", max_depth, max_states_visited, elapsed_time);
                previous_max_depth = max_depth;
                previous_max_states_visited = max_states_visited;
            }
        }

        if planning_problem.is_goal(&selected_state) {
            if verbose {
                let elapsed_time = start_time.elapsed();
                println!("max depth: {:?}, states visited: {:?}, total time: {:?}", max_depth, max_states_visited, elapsed_time);
            }
            let solution = _reconstruct_solution(&nodes, &selected_state);
            if solution.is_none() {
                println!("error in reconstructing solution")
            }
            return solution;
        }

        for action in planning_problem.applicable_actions(&selected_state) {
            let successor_state: State = planning_problem.transition_function(&selected_state, &action);
            let successor_node = nodes.get(&successor_state);
            let old_path_cost_successor_state = if successor_node.is_some() { successor_node.unwrap().path_cost } else { f64::INFINITY };
            let new_path_cost_successor_state = nodes.get(&selected_state).unwrap().path_cost + planning_problem.cost_function(&selected_state, &action, &successor_state);
            if new_path_cost_successor_state < old_path_cost_successor_state {
                nodes.insert(successor_state.clone(), Node {
                    path_cost: new_path_cost_successor_state,
                    evaluation: evaluation_function(new_path_cost_successor_state, &successor_state),
                    depth: nodes.get(&selected_state).unwrap().depth + 1,
                    parent: Some(Parent { state: selected_state.clone(), action: action.clone() }),
                });
                if !frontier_items.contains(&successor_state) {
                    frontier.push((insertion_index, successor_state.clone()), Reverse(MyF64(nodes.get(&successor_state).unwrap().evaluation)));
                    frontier_items.insert(successor_state.clone());
                    insertion_index += 1;
                }
            }
        }
    }

    if verbose {
        let elapsed_time = start_time.elapsed();
        println!("max depth: {:?}, states visited: {:?}, total time: {:?}", max_depth, max_states_visited, elapsed_time);
    }
    return None;
}

fn uniform_cost_search<State: Hash + Eq + Clone, Action: Clone>(planning_problem: &impl PlanningProblem<State, Action>, verbose: bool) -> Option<(Vec<Action>, f64)> {
    return _best_first_search(planning_problem, |path_cost, _| path_cost, verbose);
}

fn greedy_best_first_search<State: Hash + Eq + Clone, Action: Clone>(planning_problem: &impl PlanningProblem<State, Action>, heuristic_function: impl Fn(&State) -> f64, verbose: bool) -> Option<(Vec<Action>, f64)> {
    return _best_first_search(planning_problem, |_, state| heuristic_function(state), verbose);
}

fn astar<State: Hash + Eq + Clone, Action: Clone>(planning_problem: &impl PlanningProblem<State, Action>, heuristic_function: impl Fn(&State) -> f64, verbose: bool) -> Option<(Vec<Action>, f64)> {
    return _best_first_search(planning_problem, |path_cost, state| path_cost + heuristic_function(state), verbose);
}

fn weighted_astar<State: Hash + Eq + Clone, Action: Clone>(planning_problem: &impl PlanningProblem<State, Action>, heuristic_function: impl Fn(&State) -> f64, weight: f64, verbose: bool) -> Option<(Vec<Action>, f64)> {
    assert!(weight > 1.0);
    return _best_first_search(planning_problem, |path_cost, state| path_cost + weight * heuristic_function(state), verbose);
}

fn _home_problem_base(timesteps_per_hour: u32) -> HomeProblem {
    let timesteps_per_hour_f64 = timesteps_per_hour as f64;
    return HomeProblem::new(
        HomeParameters {
            horizon: 168 * timesteps_per_hour,
            battery: BatteryParameters { capacity: 3 * timesteps_per_hour, rate: 3.0 / timesteps_per_hour_f64, initial_level: 0, min_required_level: 0 },
            appliances: vec![
                ApplianceParameters { label: "washer".to_string(), duration: 2 * timesteps_per_hour, rate: 0.75 / timesteps_per_hour_f64, min_required_cycles: 3 },
                ApplianceParameters { label: "dryer".to_string(), duration: 3 * timesteps_per_hour, rate: 1.5 / timesteps_per_hour_f64, min_required_cycles: 2 },
                ApplianceParameters { label: "dishwasher".to_string(), duration: 1 * timesteps_per_hour, rate: 1.2 / timesteps_per_hour_f64, min_required_cycles: 7 },
                ApplianceParameters { label: "vehicle".to_string(), duration: 8 * timesteps_per_hour, rate: 5.0 / timesteps_per_hour_f64, min_required_cycles: 1 },
            ],
        }
    )
}

fn home_problem_1h() -> HomeProblem {
    return _home_problem_base(1);
}

fn home_problem_30m() -> HomeProblem {
    return _home_problem_base(2);
}

fn home_problem_15m() -> HomeProblem {
    return _home_problem_base(4);
}

fn main() {
    let home_problem = HomeProblem::new(
        HomeParameters {
            horizon: 9,
            battery: BatteryParameters {
                capacity: 20,
                rate: 0.4,
                initial_level: 5,
                min_required_level: 0,
            },
            appliances: vec![
                ApplianceParameters::new("washer".to_string(), 3, 0.5, 1),
                ApplianceParameters::new("dryer".to_string(), 5, 0.9, 1),
                ApplianceParameters::new("dishwasher".to_string(), 2, 0.6, 1),
                ApplianceParameters::new("vehicle".to_string(), 8, 3.75, 1),
            ],
        }
    );

    // let solution = uniform_cost_search(&home_problem, true);
    // let solution = greedy_best_first_search(&home_problem, |state| home_problem.heuristic_function(state), true);
    let solution = astar(&home_problem, |state| home_problem.heuristic_function(state), true);
    // let solution = weighted_astar(&home_problem, |state| home_problem.heuristic_function(state), 2.0, true);
    if solution.is_some() {
        let (plan, cost) = solution.unwrap();
        for action in &plan {
            print!("{:?}", action.battery);
            for appliance_action in &action.appliances {
                print!(" {:?}", appliance_action);
            }
            println!();
        }
        let min_real_cost: f64 = home_problem.min_real_cost.iter().sum();
        let real_cost = home_problem.real_cost(&plan);
        println!("cost: {:?} + {:?} = {:?}", min_real_cost, cost, real_cost);
    } else {
        println!("no solution found");
    }
}
