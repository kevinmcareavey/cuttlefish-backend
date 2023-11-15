use std::cmp::{Ordering, Reverse};
use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use std::time::Instant;
use priority_queue::PriorityQueue;

pub trait PlanningProblem<State, Action> {
    fn initial_state(&self) -> State;
    fn applicable_actions(&self, state: &State) -> Vec<&Action>;
    fn transition_function(&self, state: &State, action: &Action) -> State;
    fn cost_function(&self, state: &State, action: &Action, successor_state: &State) -> f64;
    fn is_goal(&self, state: &State) -> bool;
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

// https://github.com/garro95/priority-queue/issues/27#issuecomment-743745069
#[derive(PartialOrd, PartialEq)]
struct MyF64(f64);

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

fn reconstruct_solution<State: Hash + Eq, Action: Clone>(nodes: &HashMap<State, Node<State, Action>>, initial_state: &State, terminal_state: &State) -> Option<(Vec<Action>, f64)> {
    let mut plan = vec![];
    let mut state = terminal_state;
    loop {
        if state == initial_state {
            plan.reverse();  // actions were added in reverse order so reverse in-place
            return Some((plan, nodes.get(&terminal_state).unwrap().path_cost));
        }
        let optional_node = &nodes.get(&state);
        if optional_node.is_none() {
            break;
        }
        let optional_parent = &optional_node.unwrap().parent;
        if optional_parent.is_none() {
            break;
        }
        let parent = optional_parent.as_ref().unwrap();
        state = &parent.state;
        let action = &parent.action;
        plan.push(action.clone());
    }
    return None;
}

fn best_first_search<State: Hash + Eq + Clone, Action: Clone>(planning_problem: &impl PlanningProblem<State, Action>, evaluation_function: impl Fn(f64, &State) -> f64, verbose: bool) -> Option<(Vec<Action>, f64)> {
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
            let solution = reconstruct_solution(&nodes, &initial_state, &selected_state);
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
                let evaluation_successor_state = evaluation_function(new_path_cost_successor_state, &successor_state);
                nodes.insert(successor_state.clone(), Node {
                    path_cost: new_path_cost_successor_state,
                    evaluation: evaluation_successor_state,
                    depth: nodes.get(&selected_state).unwrap().depth + 1,
                    parent: Some(Parent { state: selected_state.clone(), action: action.clone() }),
                });
                if !frontier_items.contains(&successor_state) {
                    frontier.push((insertion_index, successor_state.clone()), Reverse(MyF64(evaluation_successor_state)));
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

pub fn uniform_cost_search<State: Hash + Eq + Clone, Action: Clone>(planning_problem: &impl PlanningProblem<State, Action>, verbose: bool) -> Option<(Vec<Action>, f64)> {
    return best_first_search(planning_problem, |path_cost, _| path_cost, verbose);
}

pub fn greedy_best_first_search<State: Hash + Eq + Clone, Action: Clone>(planning_problem: &impl PlanningProblem<State, Action>, heuristic_function: impl Fn(&State) -> f64, verbose: bool) -> Option<(Vec<Action>, f64)> {
    return best_first_search(planning_problem, |_, state| heuristic_function(state), verbose);
}

pub fn astar<State: Hash + Eq + Clone, Action: Clone>(planning_problem: &impl PlanningProblem<State, Action>, heuristic_function: impl Fn(&State) -> f64, verbose: bool) -> Option<(Vec<Action>, f64)> {
    return best_first_search(planning_problem, |path_cost, state| path_cost + heuristic_function(state), verbose);
}

pub fn weighted_astar<State: Hash + Eq + Clone, Action: Clone>(planning_problem: &impl PlanningProblem<State, Action>, heuristic_function: impl Fn(&State) -> f64, weight: f64, verbose: bool) -> Option<(Vec<Action>, f64)> {
    assert!(weight > 1.0);
    return best_first_search(planning_problem, |path_cost, state| path_cost + weight * heuristic_function(state), verbose);
}
