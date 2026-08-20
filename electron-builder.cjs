/** @type {import('electron-builder').Configuration} */
module.exports = {
  appId: "com.edifyai.desktop",
  productName: "Edify AI",
  copyright: "Copyright © Edify AI contributors",
  files: ["dist/**", "electron/**", "!**/*.map"],
  extraMetadata: { main: "electron/main.mjs" },
  directories: {
    output: "release",
    buildResources: "build-resources",
  },
  // Publish to GitHub releases so electron-updater can check for updates
  publish: {
    provider: "github",
    owner: "arcange9",
    repo: "EdifyAI",
  },
  win: {
    target: [
      { target: "nsis", arch: ["x64"] },
    ],
    artifactName: "Edify-AI-Setup-${arch}.${ext}",
  },
  nsis: {
    oneClick: false,
    perMachine: false,
    allowToChangeInstallationDirectory: true,
    createDesktopShortcut: true,
    createStartMenuShortcut: true,
    shortcutName: "Edify AI",
    // Differential update — only download the changed parts
    differentialPackage: true,
  },
};
