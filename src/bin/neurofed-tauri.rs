#![cfg_attr(
    all(not(debug_assertions), target_os = "windows"),
    windows_subsystem = "windows"
)]

#[cfg(not(feature = "web-ui"))]
fn main() {
    eprintln!(
        "This binary requires the 'web-ui' feature. Run: cargo run --features web-ui --bin neurofed-tauri"
    );
}

#[cfg(feature = "web-ui")]
fn main() -> tauri::Result<()> {
    use serde::Deserialize;
    use std::net::{SocketAddr, TcpStream};
    use std::process::Command;
    use std::time::Duration;
    use tauri::menu::{Menu, MenuItem};
    use tauri::tray::{TrayIconBuilder, TrayIconEvent};
    use tauri::{Manager, WebviewUrl, WebviewWindowBuilder};

    let url = url::Url::parse("http://localhost:8080/ui").unwrap();

    #[derive(Deserialize)]
    struct UiStateLight {
        last_source: Option<String>,
    }

    tauri::Builder::default()
        .setup(move |app| {
            // Best-effort: start backend if it's not running
            let backend_addr: SocketAddr = "127.0.0.1:8080".parse().unwrap();
            let backend_alive =
                TcpStream::connect_timeout(&backend_addr, Duration::from_millis(200)).is_ok();
            if !backend_alive {
                let mut backend_path = std::env::current_exe()?;
                #[cfg(target_os = "windows")]
                backend_path.set_file_name("neuro-fed-node.exe");
                #[cfg(not(target_os = "windows"))]
                backend_path.set_file_name("neuro-fed-node");

                let _ = Command::new(backend_path).spawn();
            }

            let window = if let Some(existing) = app.get_webview_window("main") {
                existing
            } else {
                WebviewWindowBuilder::new(app, "main", WebviewUrl::External(url.clone()))
                    .title("NeuroFed Node")
                    .inner_size(1200.0, 820.0)
                    .min_inner_size(960.0, 640.0)
                    .build()?
            };

            let _ = window.show();

            let menu = Menu::new(app)?;
            let show = MenuItem::new(app, "Show", true, None::<&str>)?;
            let hide = MenuItem::new(app, "Hide", true, None::<&str>)?;
            let open_ui = MenuItem::new(app, "Open UI", true, None::<&str>)?;
            let quit = MenuItem::new(app, "Quit", true, None::<&str>)?;
            menu.append(&show)?;
            menu.append(&hide)?;
            menu.append(&open_ui)?;
            menu.append(&quit)?;

            let icon_green =
                tauri::image::Image::from_bytes(include_bytes!("../../ui/tray_green.png"))?;
            let icon_yellow =
                tauri::image::Image::from_bytes(include_bytes!("../../ui/tray_yellow.png"))?;
            let icon_red =
                tauri::image::Image::from_bytes(include_bytes!("../../ui/tray_red.png"))?;

            let tray = TrayIconBuilder::new()
                .icon(icon_green.clone())
                .menu(&menu)
                .on_menu_event(|app, event| match event.id().as_ref() {
                    "Show" => {
                        if let Some(w) = app.get_webview_window("main") {
                            let _ = w.show();
                            let _ = w.set_focus();
                        }
                    }
                    "Hide" => {
                        if let Some(w) = app.get_webview_window("main") {
                            let _ = w.hide();
                        }
                    }
                    "Open UI" => {
                        if let Some(w) = app.get_webview_window("main") {
                            let _ = w.show();
                            let _ = w.set_focus();
                        }
                    }
                    "Quit" => {
                        app.exit(0);
                    }
                    _ => {}
                })
                .on_tray_icon_event(|tray, event| {
                    if let TrayIconEvent::Click { .. } = event {
                        let app = tray.app_handle();
                        if let Some(w) = app.get_webview_window("main") {
                            let _ = w.show();
                            let _ = w.set_focus();
                        }
                    }
                })
                .build(app)?;

            // Poll UI state and update tray color
            let tray_handle = tray.clone();
            let client = reqwest::Client::new();
            tauri::async_runtime::spawn(async move {
                let mut last_source = String::new();
                loop {
                    if let Ok(resp) = client.get("http://localhost:8080/ui/state").send().await {
                        if let Ok(state) = resp.json::<UiStateLight>().await {
                            let source =
                                state.last_source.unwrap_or_else(|| "local_pc".to_string());
                            if source != last_source {
                                let icon = match source.as_str() {
                                    "remote_llm" => &icon_red,
                                    "local_llm" => &icon_yellow,
                                    _ => &icon_green,
                                };
                                let _ = tray_handle.set_icon(Some(icon.clone()));
                                last_source = source;
                            }
                        }
                    }
                    tokio::time::sleep(Duration::from_secs(2)).await;
                }
            });

            Ok(())
        })
        .run(tauri::generate_context!())
}
