use std::collections::HashSet;
use itertools::{Itertools, izip};
use crate::basic_problem::{ApplianceAction, BatteryAction, BatteryParameters, BatteryState, HomeAction};
use crate::data::{EXPORT_PRICES, IMPORT_PRICES};
use crate::planner::{astar, greedy_best_first_search, PlanningProblem, uniform_cost_search, weighted_astar};

#[derive(Debug)]
#[derive(Clone)]
#[derive(Hash, Eq, PartialEq)]
struct ExtendedApplianceState {
    active_cycle: u32,
    completed_cycles: Vec<u32>,
}

#[derive(Debug)]
#[derive(Clone)]
#[derive(Hash, Eq, PartialEq)]
struct ExtendedHomeState {
    timestep: u32,
    battery: BatteryState,
    appliances: Vec<ExtendedApplianceState>,
}

#[derive(Debug)]
pub struct ApplianceWindowParameters {
    pub timesteps: HashSet<u32>,
    pub min_required_cycles: u32,
}

impl ApplianceWindowParameters {
    pub fn new(timesteps: HashSet<u32>, min_required_cycles: u32) -> Self {
        return Self { timesteps, min_required_cycles };
    }
}

#[derive(Debug)]
struct ExtendedApplianceParameters {
    label: String,
    duration: u32,
    rate: f64,
    min_required_cycles: Vec<ApplianceWindowParameters>,
}

impl ExtendedApplianceParameters {
    fn new(label: String, duration: u32, rate: f64, min_required_cycles: Vec<ApplianceWindowParameters>) -> Self {
        assert!(duration > 0);
        assert!(rate > 0.0);
        for (i, window_parameters_i) in min_required_cycles.iter().enumerate() {
            for (j, window_parameters_j) in min_required_cycles.iter().enumerate() {
                if i != j {
                    assert!(window_parameters_i.timesteps.is_disjoint(&window_parameters_j.timesteps));
                }
            }
        }
        return Self { label, duration, rate, min_required_cycles };
    }
}

#[derive(Debug)]
struct ExtendedHomeParameters {
    horizon: u32,
    battery: BatteryParameters,
    appliances: Vec<ExtendedApplianceParameters>,
}

#[derive(Debug)]
struct ExtendedHomeProblem {
    home_parameters: ExtendedHomeParameters,
    import_prices: Vec<f64>,
    export_prices: Vec<f64>,
    min_real_cost: Vec<f64>,
    min_required_timesteps: Vec<Vec<u32>>,
    available_actions: Vec<HomeAction>,
}

impl ExtendedHomeProblem {
    fn new(home_parameters: ExtendedHomeParameters) -> Self {
        let horizon = usize::try_from(home_parameters.horizon).unwrap();
        let import_prices = IMPORT_PRICES[0..horizon].to_vec();
        let export_prices = EXPORT_PRICES[0..horizon].to_vec();

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
            let mut window_min_required_timesteps = vec![];
            for window in &appliance_parameters.min_required_cycles {
                window_min_required_timesteps.push(&window.min_required_cycles * appliance_parameters.duration);
            }
            min_required_timesteps.push(window_min_required_timesteps);
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

        return Self { home_parameters, import_prices, export_prices, min_real_cost, min_required_timesteps, available_actions };
    }

    fn is_applicable(&self, state: &ExtendedHomeState, action: &HomeAction) -> bool {
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
            if *appliance_action == ApplianceAction::ON {
                let mut found = false;
                let required_timesteps = HashSet::from_iter((state.timestep..state.timestep + appliance_parameters.duration - appliance_state.active_cycle));
                for window_parameters in &appliance_parameters.min_required_cycles {
                    if required_timesteps.is_subset(&window_parameters.timesteps) {
                        found = true;
                        break;
                    }
                }
                if !found {
                    return false;
                }
            }
        }
        return true;
    }

    fn heuristic_function(&self, state: &ExtendedHomeState) -> f64 {
        let timesteps_to_horizon = self.home_parameters.horizon - state.timestep;

        if state.battery.level + timesteps_to_horizon < self.home_parameters.battery.min_required_level {
            return f64::INFINITY;
        }

        for (appliance_parameters, appliance_state, appliance_min_required_timesteps) in izip!(&self.home_parameters.appliances, &state.appliances, &self.min_required_timesteps) {
            for (window_parameters, window_completed_cycles, window_min_required_timesteps) in izip!(&appliance_parameters.min_required_cycles, &appliance_state.completed_cycles, appliance_min_required_timesteps) {
                let window_timesteps_completed = if window_parameters.timesteps.contains(&state.timestep) { (window_completed_cycles * appliance_parameters.duration) + appliance_state.active_cycle } else { window_completed_cycles * appliance_parameters.duration };
                if window_timesteps_completed > *window_min_required_timesteps {
                    return f64::INFINITY;
                }

                let window_timesteps_to_go = window_min_required_timesteps - window_timesteps_completed;
                if window_timesteps_to_go > timesteps_to_horizon {
                    return f64::INFINITY;
                }
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

impl PlanningProblem<ExtendedHomeState, HomeAction> for ExtendedHomeProblem {
    fn initial_state(&self) -> ExtendedHomeState {
        let mut appliance_states = vec![];
        for appliance_parameters in &self.home_parameters.appliances {
            let mut completed_cycles = vec![];
            for _ in &appliance_parameters.min_required_cycles {
                completed_cycles.push(0);
            }
            appliance_states.push(ExtendedApplianceState { active_cycle: 0, completed_cycles });
        }
        return ExtendedHomeState {
            timestep: 0,
            battery: BatteryState { level: self.home_parameters.battery.initial_level },
            appliances: appliance_states,
        };
    }

    fn applicable_actions(&self, state: &ExtendedHomeState) -> Vec<&HomeAction> {
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

    fn transition_function(&self, state: &ExtendedHomeState, action: &HomeAction) -> ExtendedHomeState {
        let successor_battery_state = BatteryState { level: if action.battery == BatteryAction::CHARGE { state.battery.level + 1 } else if action.battery == BatteryAction::DISCHARGE { state.battery.level - 1 } else { state.battery.level } };

        let mut successor_appliance_states = vec![];
        for (appliance_parameters, appliance_state, appliance_action) in izip!(&self.home_parameters.appliances, &state.appliances, &action.appliances) {
            if *appliance_action == ApplianceAction::ON {
                if appliance_state.active_cycle == appliance_parameters.duration - 1 {  // cycle ending (and possibly starting as well)
                    let mut windows = vec![];
                    for (window_parameters, window_completed_cycles) in izip!(&appliance_parameters.min_required_cycles, &appliance_state.completed_cycles) {
                        windows.push(if window_parameters.timesteps.contains(&state.timestep) { window_completed_cycles + 1 } else { *window_completed_cycles });
                    }
                    successor_appliance_states.push(ExtendedApplianceState { active_cycle: 0, completed_cycles: windows });
                } else if appliance_state.active_cycle == 0 {  // cycle starting (but not ending)
                    successor_appliance_states.push(ExtendedApplianceState { active_cycle: 1, completed_cycles: appliance_state.completed_cycles.clone() });
                } else {
                    successor_appliance_states.push(ExtendedApplianceState { active_cycle: appliance_state.active_cycle + 1, completed_cycles: appliance_state.completed_cycles.clone() });
                }
            } else {
                successor_appliance_states.push(ExtendedApplianceState { active_cycle: 0, completed_cycles: appliance_state.completed_cycles.clone() });
            }
        }

        return ExtendedHomeState {
            timestep: state.timestep + 1,
            battery: successor_battery_state,
            appliances: successor_appliance_states,
        };
    }

    fn cost_function(&self, state: &ExtendedHomeState, action: &HomeAction, _successor_state: &ExtendedHomeState) -> f64 {
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

    fn is_goal(&self, state: &ExtendedHomeState) -> bool {
        if state.timestep != self.home_parameters.horizon {
            return false;
        }
        if state.battery.level < self.home_parameters.battery.min_required_level {
            return false;
        }
        for (appliance_parameters, appliance_state) in izip!(&self.home_parameters.appliances, &state.appliances) {
            for (window_parameters, window_completed_cycles) in izip!(&appliance_parameters.min_required_cycles, &appliance_state.completed_cycles) {
                if window_completed_cycles < &window_parameters.min_required_cycles {
                    return false;
                }
            }
        }
        return true;
    }
}

fn home_problem_toy(horizon: u32) -> ExtendedHomeProblem {
    let timesteps: HashSet<u32> = HashSet::from_iter(0..horizon);
    return ExtendedHomeProblem::new(
        ExtendedHomeParameters {
            horizon,
            battery: BatteryParameters {
                capacity: 20,
                rate: 0.4,
                initial_level: 5,
                min_required_level: 0,
            },
            appliances: vec![
                ExtendedApplianceParameters::new("washer".to_string(), 3, 0.5, vec![ApplianceWindowParameters::new(timesteps.clone(), 1)]),
                ExtendedApplianceParameters::new("dryer".to_string(), 5, 0.9, vec![ApplianceWindowParameters::new(timesteps.clone(), 1)]),
                ExtendedApplianceParameters::new("dishwasher".to_string(), 2, 0.6, vec![ApplianceWindowParameters::new(timesteps.clone(), 1)]),
                ExtendedApplianceParameters::new("vehicle".to_string(), 8, 3.75, vec![ApplianceWindowParameters::new(timesteps.clone(), 1)]),
            ],
        }
    );
}

fn home_problem_basic(timesteps_per_hour: u32) -> ExtendedHomeProblem {
    assert!(1 <= timesteps_per_hour && timesteps_per_hour <= 60 && 60 % timesteps_per_hour == 0);
    let days: u32 = 7;
    let timesteps_per_day = 24 * timesteps_per_hour;
    let horizon = days * timesteps_per_day;
    let timesteps: HashSet<u32> = (0..horizon).collect();
    let timesteps_per_hour_f64 = timesteps_per_hour as f64;
    return ExtendedHomeProblem::new(
        ExtendedHomeParameters {
            horizon: 168 * timesteps_per_hour,
            battery: BatteryParameters { capacity: 3 * timesteps_per_hour, rate: 3.0 / timesteps_per_hour_f64, initial_level: 0, min_required_level: 0 },
            appliances: vec![
                ExtendedApplianceParameters::new("washer".to_string(), 2 * timesteps_per_hour, 0.75 / timesteps_per_hour_f64, vec![ApplianceWindowParameters::new(timesteps.clone(), 3)]),
                ExtendedApplianceParameters::new("dryer".to_string(), 3 * timesteps_per_hour, 1.5 / timesteps_per_hour_f64, vec![ApplianceWindowParameters::new(timesteps.clone(), 2)]),
                ExtendedApplianceParameters::new("dishwasher".to_string(), 1 * timesteps_per_hour, 1.2 / timesteps_per_hour_f64, vec![ApplianceWindowParameters::new(timesteps.clone(), 7)]),
                ExtendedApplianceParameters::new("vehicle".to_string(), 8 * timesteps_per_hour, 5.0 / timesteps_per_hour_f64, vec![ApplianceWindowParameters::new(timesteps.clone(), 1)]),
            ],
        }
    )
}

fn home_problem_extended(timesteps_per_hour: u32) -> ExtendedHomeProblem {
    assert!(1 <= timesteps_per_hour && timesteps_per_hour <= 60 && 60 % timesteps_per_hour == 0);
    let days: u32 = 7;
    let timesteps_per_day = 24 * timesteps_per_hour;
    let horizon = days * timesteps_per_day;
    let washer_dryer_timesteps = HashSet::from_iter((0..horizon).filter(|timestep| 9 <= timestep % timesteps_per_day && timestep % timesteps_per_day < 20));  // any day between 09:00 and 20:00
    let mut dishwasher_windows = vec![];  // each day between 19:00 and 23:00
    for day in 0..days {
        let mut day_dishwasher_timesteps = HashSet::new();
        for day_timestep in 0..timesteps_per_day {
            if 19 <= day_timestep && day_timestep < 23 {
                day_dishwasher_timesteps.insert((day * timesteps_per_day) + day_timestep);
            }
        }
        dishwasher_windows.push(ApplianceWindowParameters::new(day_dishwasher_timesteps.clone(), 1));
    }
    let vehicle_timesteps = HashSet::from_iter((0..horizon).filter(|timestep| timestep >= &(5 * timesteps_per_day)));  // Saturday or Sunday
    let timesteps_per_hour_f64 = timesteps_per_hour as f64;
    return ExtendedHomeProblem::new(
        ExtendedHomeParameters {
            horizon: 168 * timesteps_per_hour,
            battery: BatteryParameters { capacity: 3 * timesteps_per_hour, rate: 3.0 / timesteps_per_hour_f64, initial_level: 0, min_required_level: 0 },
            appliances: vec![
                ExtendedApplianceParameters::new("washer".to_string(), 2 * timesteps_per_hour, 0.75 / timesteps_per_hour_f64, vec![ApplianceWindowParameters::new(washer_dryer_timesteps.clone(), 3)]),
                ExtendedApplianceParameters::new("dryer".to_string(), 3 * timesteps_per_hour, 1.5 / timesteps_per_hour_f64, vec![ApplianceWindowParameters::new(washer_dryer_timesteps.clone(), 2)]),
                ExtendedApplianceParameters::new("dishwasher".to_string(), 1 * timesteps_per_hour, 1.2 / timesteps_per_hour_f64, dishwasher_windows),
                ExtendedApplianceParameters::new("vehicle".to_string(), 8 * timesteps_per_hour, 5.0 / timesteps_per_hour_f64, vec![ApplianceWindowParameters::new(vehicle_timesteps.clone(), 1)]),
            ],
        }
    )
}

pub fn run() {
    // let home_problem = home_problem_toy(9);  // states visited: 7597, total time: 66.206ms, cost: -16.564 + 271.8064 = 255.2424
    // let home_problem = home_problem_basic(1);  // states visited: 3618227, total time: 53.979608959s, cost: -2988.1500000000024 + 3296.974650000001 = 308.8246500000001
    let home_problem = home_problem_extended(1);  // states visited: 2924447, total time: 36.957981542s, cost: -2988.1500000000024 + 3324.2307000000023 = 336.0807000000001

    // let solution = uniform_cost_search(&home_problem, true);
    let solution = astar(&home_problem, |state| home_problem.heuristic_function(state), true);
    // let solution = weighted_astar(&home_problem, |state| home_problem.heuristic_function(state), 2.0, true);
    // let solution = greedy_best_first_search(&home_problem, |state| home_problem.heuristic_function(state), true);
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
