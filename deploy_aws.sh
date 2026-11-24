#!/bin/bash
# ==============================================================================
# Script de Despliegue Automático en AWS EC2 (Docker + FastAPI + MLOps)
# 100% Free Tier Garantizado en Ohio (us-east-2) o cualquier región activa
# Auto-detecta si la cuenta requiere t4g.micro (ARM64) o t3/t2.micro (x86_64)
# ==============================================================================
set -e

# Detectar región activa de AWS CLI / CloudShell
REGION=$(aws configure get region 2>/dev/null || echo "")
if [ -z "$REGION" ]; then
    REGION="${AWS_REGION:-us-east-2}"
fi

echo "=================================================================="
echo "🚀 INICIANDO DESPLIEGUE EN AWS ($REGION - 100% FREE TIER)"
echo "=================================================================="

# 1. Detectar el tipo de instancia Free Tier permitido por la política de la cuenta
echo "[1/5] Detectando tipo de instancia Free Tier autorizado..."
FREE_TYPES=$(aws ec2 describe-instance-types --region "$REGION" --filters "Name=free-tier-eligible,Values=true" --query "InstanceTypes[*].InstanceType" --output text 2>/dev/null || true)

if echo "$FREE_TYPES" | grep -w -q "t4g.micro"; then
    INSTANCE_TYPE="t4g.micro"
    ARCH="arm64"
elif echo "$FREE_TYPES" | grep -w -q "t3.micro"; then
    INSTANCE_TYPE="t3.micro"
    ARCH="x86_64"
elif echo "$FREE_TYPES" | grep -w -q "t2.micro"; then
    INSTANCE_TYPE="t2.micro"
    ARCH="x86_64"
else
    EXISTING_TYPE=$(aws ec2 describe-instances --region "$REGION" --filters "Name=instance-state-name,Values=running" --query "Reservations[0].Instances[0].InstanceType" --output text 2>/dev/null || true)
    if [ -n "$EXISTING_TYPE" ] && [ "$EXISTING_TYPE" != "None" ]; then
        INSTANCE_TYPE="$EXISTING_TYPE"
    else
        INSTANCE_TYPE="t4g.micro"
    fi
    if [[ "$INSTANCE_TYPE" == t4g* ]]; then
        ARCH="arm64"
    else
        ARCH="x86_64"
    fi
fi
echo "      Instancia seleccionada : $INSTANCE_TYPE ($ARCH)"

# 2. Seleccionar AMI oficial Amazon Linux 2023 compatible con la arquitectura
echo "[2/5] Buscando AMI Amazon Linux 2023 ($ARCH)..."
AMI_ID=$(aws ec2 describe-images \
    --region "$REGION" \
    --owners amazon \
    --filters "Name=name,Values=al2023-ami-2023.*-$ARCH" "Name=state,Values=available" "Name=architecture,Values=$ARCH" \
    --query "sort_by(Images, &CreationDate)[-1].ImageId" \
    --output text)
ROOT_DEV=$(aws ec2 describe-images --region "$REGION" --image-ids "$AMI_ID" --query "Images[0].RootDeviceName" --output text 2>/dev/null || echo "/dev/xvda")
echo "      AMI oficial : $AMI_ID (Root: $ROOT_DEV)"

# 3. Red y Grupo de Seguridad
echo "[3/5] Verificando Red VPC y Security Group..."
VPC_ID=$(aws ec2 describe-vpcs --region "$REGION" --filters "Name=isDefault,Values=true" --query "Vpcs[0].VpcId" --output text)
SUBNET_ID=$(aws ec2 describe-subnets --region "$REGION" --filters "Name=vpc-id,Values=$VPC_ID" --query "Subnets[0].SubnetId" --output text | awk '{print $1}')

SG_NAME="oee-pisa-sg"
SG_ID=$(aws ec2 describe-security-groups --region "$REGION" --filters "Name=group-name,Values=$SG_NAME" "Name=vpc-id,Values=$VPC_ID" --query "SecurityGroups[0].GroupId" --output text 2>/dev/null || true)
if [ -z "$SG_ID" ] || [ "$SG_ID" == "None" ]; then
    echo "      Creando Security Group $SG_NAME..."
    SG_ID=$(aws ec2 create-security-group --region "$REGION" --group-name "$SG_NAME" --description "SG OEE PiSA" --vpc-id "$VPC_ID" --query "GroupId" --output text)
fi
# Asegurar puertos 80, 8000 y 22
aws ec2 authorize-security-group-ingress --region "$REGION" --group-id "$SG_ID" --protocol tcp --port 80 --cidr 0.0.0.0/0 2>/dev/null || true
aws ec2 authorize-security-group-ingress --region "$REGION" --group-id "$SG_ID" --protocol tcp --port 8000 --cidr 0.0.0.0/0 2>/dev/null || true
aws ec2 authorize-security-group-ingress --region "$REGION" --group-id "$SG_ID" --protocol tcp --port 22 --cidr 0.0.0.0/0 2>/dev/null || true
echo "      Security Group listo : $SG_ID (Puertos 80, 8000, 22 abiertos)"

# 4. UserData con Swap de 2GB y Docker
cat << 'USERDATA_EOF' > /tmp/oee_boot.sh
#!/bin/bash
dd if=/dev/zero of=/swapfile bs=1M count=2048
chmod 600 /swapfile
mkswap /swapfile
swapon /swapfile
echo "/swapfile swap swap defaults 0 0" >> /etc/fstab
dnf update -y
dnf install -y docker git
systemctl start docker
systemctl enable docker
mkdir -p /opt/oee-app && cd /opt/oee-app
git clone https://github.com/SalmiBC/26_proyecto_OEE.git .
docker build -t oee-api .
docker run -d --restart always -p 80:8000 -p 8000:8000 --name oee-service oee-api
USERDATA_EOF

# 5. Lanzar la nueva instancia EC2 dedicada
echo "[4/5] Lanzando nueva instancia $INSTANCE_TYPE en $REGION..."
NEW_INSTANCE_ID=$(aws ec2 run-instances \
    --region "$REGION" \
    --image-id "$AMI_ID" \
    --instance-type "$INSTANCE_TYPE" \
    --security-group-ids "$SG_ID" \
    --subnet-id "$SUBNET_ID" \
    --block-device-mappings "[{\"DeviceName\":\"$ROOT_DEV\",\"Ebs\":{\"VolumeSize\":15,\"VolumeType\":\"gp3\",\"DeleteOnTermination\":true}}]" \
    --user-data file:///tmp/oee_boot.sh \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=oee-predictivo-pisa}]' \
    --query "Instances[0].InstanceId" \
    --output text)

if [ -z "$NEW_INSTANCE_ID" ] || [ "$NEW_INSTANCE_ID" == "None" ]; then
    echo "❌ Error al crear la instancia EC2 en AWS."
    exit 1
fi

echo "[5/5] Instancia EC2 creada con ID: $NEW_INSTANCE_ID"
echo "      Esperando estado 'running' y asignación de IP pública..."
aws ec2 wait instance-running --region "$REGION" --instance-ids "$NEW_INSTANCE_ID"

NEW_IP=$(aws ec2 describe-instances --region "$REGION" --instance-ids "$NEW_INSTANCE_ID" --query "Reservations[0].Instances[0].PublicIpAddress" --output text)

echo ""
echo "=================================================================="
echo "🎉 ¡NUEVA INSTANCIA OEE EN AWS DESPLEGADA EXITOSAMENTE!"
echo "=================================================================="
echo "Región         : $REGION (Ohio)"
echo "Instancia ID   : $NEW_INSTANCE_ID ($INSTANCE_TYPE - Free Tier)"
echo "NUEVA IP       : $NEW_IP"
echo ""
echo "En ~2 a 3 minutos (mientras Docker compila en segundo plano) abre:"
echo "👉 Aplicación Web     : http://$NEW_IP"
echo "👉 Swagger OpenAPI   : http://$NEW_IP/docs"
echo "👉 Rúbrica Demo      : http://$NEW_IP/presentacion"
echo "👉 Salud de la API   : http://$NEW_IP/health"
echo "=================================================================="
