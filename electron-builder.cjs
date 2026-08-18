/** @type {import('electron-builder').Configuration} */
module.exports = {
  appId: "com.edifyai.desktop",
  productName: "Edify AI",
  copyright: "Copyright © Edify AI",
  publish: null,
  files: ["dist/**", "electron/**", "!**/*.map"],
  extraMetadata: { main: "electron/main.mjs" },
  directories: {
    output: "release",
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
  },
  portable: {
    artifactName: "Edify-AI-Portable-${arch}.${ext}",
  },
};
