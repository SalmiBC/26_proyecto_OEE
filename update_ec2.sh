#!/bin/bash
# ==============================================================================
# Script de Actualización Rápida en AWS EC2 sin reiniciar la máquina
# ==============================================================================
set -e

REGION=$(aws configure get region 2>/dev/null || echo "us-east-2")
REGION="${REGION:-us-east-2}"

echo "🔄 Buscando instancia activa OEE en AWS ($REGION)..."
INSTANCE_ID=$(aws ec2 describe-instances --region "$REGION" \
    --filters "Name=tag:Name,Values=oee-predictivo-pisa" "Name=instance-state-name,Values=running" \
    --query "Reservations[0].Instances[0].InstanceId" --output text 2>/dev/null || true)

if [ -z "$INSTANCE_ID" ] || [ "$INSTANCE_ID" == "None" ]; then
    INSTANCE_ID="i-0056f9a496e3b5086"
fi

PUBLIC_IP=$(aws ec2 describe-instances --region "$REGION" --instance-ids "$INSTANCE_ID" \
    --query "Reservations[0].Instances[0].PublicIpAddress" --output text 2>/dev/null || echo "13.59.96.134")

echo "Instancia encontrada: $INSTANCE_ID ($PUBLIC_IP)"

# Generar llave SSH temporal
rm -f /tmp/ec2_temp_key*
ssh-keygen -t rsa -f /tmp/ec2_temp_key -N "" -q

echo "Enviando llave efímera a EC2 Instance Connect..."
aws ec2-instance-connect send-ssh-public-key \
    --region "$REGION" \
    --instance-id "$INSTANCE_ID" \
    --instance-os-user ec2-user \
    --ssh-public-key file:///tmp/ec2_temp_key.pub > /dev/null

echo "Sincronizando cambios en el contenedor Docker en vivo..."
ssh -i /tmp/ec2_temp_key -o StrictHostKeyChecking=no -o ConnectTimeout=10 ec2-user@"$PUBLIC_IP" << 'REMOTE_EOF'
cd /opt/oee-app
sudo git pull origin main
sudo docker cp app_consulta.html oee-service:/app/app_consulta.html
sudo docker cp presentacion_rubrica_oee.html oee-service:/app/presentacion_rubrica_oee.html 2>/dev/null || true
sudo docker cp dashboard_oee_predictivo.html oee-service:/app/dashboard_oee_predictivo.html 2>/dev/null || true
echo "Contenedor Docker actualizado con éxito."
REMOTE_EOF

echo ""
echo "=================================================================="
echo "✅ ¡ACTUALIZACIÓN COMPLETADA CON ÉXITO!"
echo "Refresca tu navegador en: http://$PUBLIC_IP"
echo "=================================================================="
