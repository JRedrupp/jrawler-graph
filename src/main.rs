use std::{
    collections::{HashMap, HashSet, VecDeque}, sync::Arc, time::Duration
};

use clap::Parser; // Add Clap for command-line argument parsing
use log::{error, info, warn, debug}; // Import logging macros
use petgraph::{
    graph::{NodeIndex, UnGraph},
    visit::EdgeRef,
    dot::Dot,
}; // Using UnGraph for simplicity, could use DiGraph
use reqwest::Client;
use scraper::{Html, Selector};
use url::Url;
use std::fs;
use regex::Regex; // Add Regex for pattern matching
use pathfinding::directed::dijkstra::dijkstra;


// Define a custom error type for cleaner error handling
#[derive(thiserror::Error, Debug)]
enum CrawlerError {
    #[error("Network request failed: {0}")]
    RequestError(#[from] reqwest::Error),
    #[error("URL parsing failed: {0}")]
    UrlParseError(#[from] url::ParseError),
    #[error("Invalid base URL")]
    InvalidBaseUrl,
    #[error("Missing host in URL: {0}")]
    MissingHost(String),
    #[error("I/O Error: {0}")]
    IoError(#[from] std::io::Error), // Added for logger init
}

/// Command-line arguments for the crawler
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Starting URL for the crawl
    #[arg(short, long)]
    start: String,

    /// Target URL to find the shortest path to
    #[arg(short, long)]
    target: String,

    /// Maximum number of pages to crawl
    #[arg(short, long, default_value_t = 5)]
    max_pages: usize,

    /// Regex patterns to ignore (comma-separated)
    #[arg(short, long, default_value = "")]
    ignore_patterns: String,
}

fn save_graph_as_svg(graph: &UnGraph<String, ()>, output_path: &str) -> Result<(), CrawlerError> {
    // Generate DOT representation of the graph
    let dot = Dot::new(&graph);
    let dot_representation = format!("{:?}", dot);

    // Save the DOT file
    let dot_file_path = format!("{}.dot", output_path);
    fs::write(&dot_file_path, &dot_representation)?;

    // Use `dot` command-line tool to convert DOT to SVG
    let output = std::process::Command::new("dot")
        .arg("-Tsvg")
        .arg(dot_file_path)
        .arg("-o")
        .arg(output_path)
        .output();

    match output {
        Ok(_) => {
            println!("Graph saved as SVG at: {}", output_path);
            Ok(())
        }
        Err(e) => {
            error!("Failed to generate SVG: {}", e);
            Err(CrawlerError::IoError(e))
        }
    }
}

/// Processes a single URL: fetches, parses, and extracts links.
async fn process_url(
    client: Arc<Client>,
    current_url: Url,
    start_domain: &str,
    ignore_regexes: &[Regex],
    graph: &mut UnGraph<String, ()>,
    url_to_node: &mut HashMap<Url, NodeIndex>,
    visited: &mut HashSet<Url>,
    queue: &mut VecDeque<Url>,
) -> Result<(), CrawlerError> {
    let normalized_current_url = normalize_url(current_url.clone());
    let current_node = match url_to_node.get(&normalized_current_url) {
        Some(node) => *node,
        None => {
            warn!("URL {} not found in url_to_node map, skipping.", current_url);
            return Ok(());
        }
    };

    // Fetch and parse the page
    match fetch_and_parse(&client, &current_url).await {
        Ok(Some(document)) => {
            // Extract links
            let links = extract_links(
                &document,
                &current_url,
                start_domain,
                &[".php"],
                ignore_regexes,
                visited,
            )?;

            debug!("Found {} valid links on {}", links.len(), current_url);

            for link_url in links {
                let normalized_link_url = normalize_url(link_url.clone());

                // Add node and edge if the link is new
                let target_node = *url_to_node.entry(normalized_link_url.clone()).or_insert_with(|| {
                    debug!("Adding new node for: {}", link_url);
                    graph.add_node(link_url.to_string())
                });
                
                // Add edge between current page and linked page
                if current_node != target_node {
                    graph.update_edge(current_node, target_node, ());
                }

                // If the link hasn't been visited and is within the domain, add to queue
                if visited.insert(normalized_link_url.clone()) {
                    debug!("Queueing new URL: {}", link_url);
                    queue.push_back(link_url);
                } else {
                    debug!("Already visited or queued: {}", link_url);
                }
            }
        }
        Ok(None) => {
            warn!("Skipping non-HTML or inaccessible URL: {}", current_url);
        }
        Err(e) => {
            error!("Failed to process {}: {}", current_url, e);
        }
    }

    Ok(())
}

/// Crawls URLs starting from the initial queue.
async fn crawl(
    client: Arc<Client>,
    start_domain: &str,
    ignore_regexes: &[Regex],
    max_pages: usize,
    queue: &mut VecDeque<Url>,
    visited: &mut HashSet<Url>,
    graph: &mut UnGraph<String, ()>,
    url_to_node: &mut HashMap<Url, NodeIndex>,
) -> Result<(), CrawlerError> {
    let mut pages_crawled = 0;

    while let Some(current_url) = queue.pop_front() {
        if pages_crawled >= max_pages {
            info!("Reached maximum page limit ({})", max_pages);
            break;
        }

        info!(
            "Crawling [{} / {}]: {}",
            pages_crawled + 1,
            max_pages,
            current_url
        );

        process_url(
            client.clone(),
            current_url,
            start_domain,
            ignore_regexes,
            graph,
            url_to_node,
            visited,
            queue,
        )
        .await?;

        pages_crawled += 1;
    }

    info!("Crawling finished. Visited {} pages.", pages_crawled);
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), CrawlerError> {
    // Parse command-line arguments
    let args = Args::parse();

    // Initialize the logger
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    // Compile regex patterns to ignore
    let ignore_regexes: Vec<Regex> = args
        .ignore_patterns
        .split(',')
        .filter(|pattern| !pattern.is_empty())
        .map(|pattern| Regex::new(pattern).expect("Invalid regex pattern"))
        .collect();

    // --- Configuration ---
    let start_url_str = &args.start;
    let target_url_str = &args.target;
    let max_pages = args.max_pages;
    // ---------------------

    let start_url = Url::parse(start_url_str)?;
    let target_url = Url::parse(target_url_str)?;
    let start_domain = start_url
        .host_str()
        .ok_or_else(|| CrawlerError::MissingHost(start_url_str.to_string()))?
        .to_string();

    info!("Starting crawl at: {}", start_url);
    info!("Restricting to domain: {}", start_domain);
    info!("Maximum pages to crawl: {}", max_pages);

    // Use a shared client for connection pooling
    let client = Arc::new(
        Client::builder()
            .timeout(Duration::from_secs(10)) // Add a request timeout
            .user_agent("SimpleRustCrawler/1.0") // Set a user agent
            .build()?,
    );

    // Data structures for crawling
    let mut queue: VecDeque<Url> = VecDeque::new(); // URLs to visit
    let mut visited: HashSet<Url> = HashSet::new(); // Visited URLs
    let mut graph: UnGraph<String, ()> = UnGraph::new_undirected(); // Page graph (URL string, no edge data)
    let mut url_to_node: HashMap<Url, NodeIndex> = HashMap::new(); // Map URL to graph node index

    // Initialize with the start URL
    queue.push_back(start_url.clone());
    visited.insert(normalize_url(start_url.clone())); // Store normalized URL in visited set
    let start_node = graph.add_node(start_url.to_string());
    url_to_node.insert(normalize_url(start_url.clone()), start_node); // Map normalized URL

    // Perform the crawl
    crawl(
        client,
        &start_domain,
        &ignore_regexes,
        max_pages,
        &mut queue,
        &mut visited,
        &mut graph,
        &mut url_to_node,
    )
    .await?;
    
    // --- Find the shortest path to the target URL ---
    if let Some(&target_node) = url_to_node.get(&normalize_url(target_url.clone())) {
        let start_node = url_to_node
            .get(&normalize_url(start_url.clone()))
            .ok_or(CrawlerError::InvalidBaseUrl)?;

        if let Some((path, cost)) = dijkstra(
            start_node,
            |node| graph.edges(*node).map(|edge| (edge.target(), 1)),
            |node| *node == target_node,
        ) {
            println!("\n--- Shortest Path to Target ---");
            println!("Cost: {}", cost);

            for node in &path {
            println!("  - {}", graph[*node]);
            }
            println!("-------------------------------\n");

            // Create a subgraph for the shortest path
            let mut path_graph = UnGraph::new_undirected();
            let mut path_nodes = HashMap::new();

            for node in &path {
            let node_index = path_graph.add_node(graph[*node].clone());
            path_nodes.insert(*node, node_index);
            }

            for window in path.windows(2) {
            if let [source, target] = window {
                path_graph.add_edge(
                *path_nodes.get(source).unwrap(),
                *path_nodes.get(target).unwrap(),
                (),
                );
            }
            }

            // Save the shortest path graph as SVG
            save_graph_as_svg(&path_graph, "shortest_path.svg")?;
        } else {
            println!("No path found to the target URL: {}", target_url);
        }
        } else {
        println!("Target URL not found in the graph: {}", target_url);
        }

    Ok(())
}

/// Fetches a URL, checks if it's HTML, and parses it.
async fn fetch_and_parse(client: &Arc<Client>, url: &Url) -> Result<Option<Html>, CrawlerError> {
    debug!("Fetching: {}", url);
    let response = client.get(url.clone()).send().await?; // Clone URL for request

    // Check status code
    if !response.status().is_success() {
        warn!("Received non-success status {} for {}", response.status(), url);
        return Ok(None); // Treat non-success as skippable
    }

    // Check content type
    let content_type = response
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|val| val.to_str().ok())
        .unwrap_or("");

    if !content_type.contains("text/html") {
        debug!("Skipping non-HTML content type '{}' for {}", content_type, url);
        return Ok(None);
    }

    // Read body and parse HTML
    let body = response.text().await?;
    debug!("Parsing HTML for {}", url);
    let document = Html::parse_document(&body);
    Ok(Some(document))
}

/// Extracts valid, same-domain HTTP/HTTPS links from an HTML document.
fn extract_links(
    document: &Html,
    base_url: &Url,
    allowed_domain: &str,
    restricted_extensions: &[&str],
    ignore_regexes: &[Regex],
    visited: &mut HashSet<Url>
) -> Result<Vec<Url>, CrawlerError> {
    let link_selector = Selector::parse("a[href]").expect("Invalid CSS selector for links");
    let mut valid_links =   HashSet::new();

    for element in document.select(&link_selector) {
        if let Some(href) = element.value().attr("href") {
            
            if href.starts_with("#") {
                // Ignore fragment links
                continue;
            }

            // Attempt to parse the href relative to the base URL
            match base_url.join(href) {
                Ok(mut potential_url) => {
                    
                    // Only consider http and https schemes
                    if potential_url.scheme() == "http" || potential_url.scheme() == "https" {
                        // Check if the link is within the allowed domain
                        if let Some(host) = potential_url.host_str() {
                            if host == allowed_domain || host.ends_with(&format!(".{}", allowed_domain)) {
                                // Check for restricted extensions
                                if restricted_extensions.iter().any(|ext| potential_url.path().ends_with(ext)) {
                                    debug!("Ignoring link with restricted extension: {}", potential_url);
                                    continue;
                                }

                                // Check against ignore regex patterns
                                if ignore_regexes.iter().any(|regex| regex.is_match(potential_url.as_str())) {
                                    debug!("Ignoring link matching ignore pattern: {}", potential_url);
                                    continue;
                                }
                                
                                // Normalize URL (remove fragment) before adding
                                potential_url.set_fragment(None);
                                
                                // Check if the link is already visited
                                if visited.contains(&potential_url) {
                                    debug!("Ignoring already visited link: {}", potential_url);
                                    continue;
                                }

                                if &potential_url == base_url {
                                    debug!("Ignoring link to the same URL: {}", potential_url);
                                    continue;
                                }

                                valid_links.insert(potential_url.clone());
                            } else {
                                debug!("Ignoring link to different domain: {}", potential_url);
                            }
                        } else {
                            debug!("Ignoring link with no host: {}", potential_url);
                        }
                    } else {
                        debug!("Ignoring non-HTTP(S) link: {}", potential_url);
                    }
                }
                Err(e) => {
                    // Log parsing errors for individual hrefs but don't stop the crawl
                    warn!("Could not parse href '{}' relative to {}: {}", href, base_url, e);
                }
            }
        }
    }
    Ok(valid_links.into_iter().collect())
}

/// Normalizes a URL for consistent storage and comparison (e.g., removes fragment).
fn normalize_url(mut url: Url) -> Url {
    url.set_fragment(None);
    // Potentially add more normalization: remove trailing slash, lowercase domain, etc.
    url
}