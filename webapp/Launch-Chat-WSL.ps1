# Ejecuta el entorno web dentro de WSL llamando a un único script
$WSLScript = "/home/carlos/RAGExpoArt/webapp/environment.sh"

# Si quieres especificar distro (opcional), descomenta la siguiente línea:
# $Distro = "Ubuntu-22.04"

# Lanza el script en WSL
if ($PSBoundParameters.ContainsKey('Distro') -and $Distro) {
  wsl -d $Distro -- bash -lc "bash '$WSLScript'"
} else {
  wsl -- bash -lc "bash '$WSLScript'"
}

# (Opcional) abre el navegador automáticamente
Start-Sleep -Seconds 2
Start-Process "http://localhost:8000"

