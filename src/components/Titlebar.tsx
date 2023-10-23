import * as React from "react";
import { platform } from "@tauri-apps/plugin-os";
import { WindowTitlebar } from "tauri-controls";

export function Titlebar(props: any) {
    const [controlsOrder, setControlsOrder] = React.useState<"left" | "right">("right");
    const [osPlatform, setOsPlatform] = React.useState<"macos" | "windows">("windows");
    const [controlStyle, setControlStyle] = React.useState<"" | "p-2">("p-2");
    
    // automatic os detection doesn't work properly for some reason, so doing it manually
    React.useEffect(() => {
        async function fetchPlatform() {
            const platformName = await platform();
            if (platformName === "darwin") {
                setControlsOrder("left");
                setOsPlatform("macos");
                setControlStyle("p-2")
            } else {
                setControlsOrder("right");
                setOsPlatform("windows");
                setControlStyle("")
            }
        }

        fetchPlatform();
    }, []);

    return (
        <WindowTitlebar
            // make controls invisible on macOS and use native ones, but keep dragging properties intact
            style={{ opacity: osPlatform === "macos" ? "0%" : "100%" }}
            controlsOrder={controlsOrder}
            windowControlsProps={{
                platform: osPlatform
            }}
            className={controlStyle}
        >
        </WindowTitlebar>
    );
}